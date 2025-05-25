"""
Blockchain Connector for direct interaction with blockchain data.
This module provides functionality to fetch transaction data, events, and smart contract states
directly from the blockchain, replacing the static JSON file approach.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from web3 import Web3
from web3.middleware import geth_poa_middleware
import time
from dotenv import load_dotenv
import numpy as np

# JSON encoder tùy chỉnh để xử lý kiểu dữ liệu NumPy
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder hỗ trợ các kiểu dữ liệu NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logging.info(f"Loaded environment variables from {dotenv_path}")
else:
    logging.warning(f"No .env file found at {dotenv_path}")
    
# Alternative paths for .env
alt_dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "w3storage-upload-script/ifps_qr.env")
if os.path.exists(alt_dotenv_path):
    with open(alt_dotenv_path, 'r') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('//'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    logging.info(f"Loaded environment variables from {alt_dotenv_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../fl_integration_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("blockchain_connector")

class BlockchainConnector:
    """
    Connector class for interacting with blockchain data.
    Provides methods to fetch transaction data, events, and smart contract states.
    """
    
    def __init__(self, 
                 rpc_url: str = None, 
                 contract_address: str = None, 
                 contract_abi_path: str = None,
                 cache_dir: str = "./cache"):
        """
        Initialize the blockchain connector.
        
        Args:
            rpc_url: RPC URL for the blockchain network
            contract_address: Address of the smart contract
            contract_abi_path: Path to the contract ABI file
            cache_dir: Directory to store cache files
        """
        # Always set cache_dir at the very top
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Use environment variables if parameters are not provided
        self.rpc_url = rpc_url or os.getenv("POLYGON_AMOY_RPC", "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94")
        self.contract_address = contract_address or os.getenv("CONTRACT_ADDRESS", "0x88b7eb798735ce65eb282df165e01e30de8786e3")
        self.private_key = os.getenv("PRIVATE_KEY", "5d014eee1544fe8e166b0ffc5d9a5da41456cec5d4c15b76611e8747d848f079")
        
        # Tìm ABI từ đường dẫn hợp lệ 
        if contract_abi_path:
            self.contract_abi_path = contract_abi_path
        else:
            # Thử với đường dẫn trong config
            potential_paths = [
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "SupplyChain_dapp/artifacts/contracts/SupplyChainNFT.sol/SupplyChainNFT.json"
                ),
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "SupplyChain_dapp/artifacts/contracts/SupplyChain.sol/SupplyChain.json"
                ),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "abi/SupplyChain.json"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "abi/SupplyChainNFT.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "abi/SupplyChain.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "abi/Contract.json")
            ]
            
            self.contract_abi_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    self.contract_abi_path = path
                    logger.info(f"Found ABI file at {path}")
                    break
            
            # Nếu không tìm thấy, thử tạo ABI tối thiểu để test kết nối
            if not self.contract_abi_path:
                logger.warning(f"No ABI file found in potential paths, creating minimal ABI")
                minimal_abi_dir = os.path.join(self.cache_dir, "abi")
                os.makedirs(minimal_abi_dir, exist_ok=True)
                minimal_abi_path = os.path.join(minimal_abi_dir, "MinimalContract.json")
                
                # ABI tối thiểu để test kết nối
                minimal_abi = [
                    {
                        "inputs": [],
                        "name": "getAllNodes",
                        "outputs": [{"type": "address[]", "name": ""}],
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
                
                try:
                    with open(minimal_abi_path, 'w') as f:
                        json.dump({"abi": minimal_abi}, f)
                    self.contract_abi_path = minimal_abi_path
                except Exception as e:
                    logger.error(f"Failed to create minimal ABI: {str(e)}")
                    self.contract_abi_path = potential_paths[0]
        
        # Create cache directory if it doesn't exist
        # self.cache_dir = cache_dir
        # os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Web3 connection
        self.w3 = None
        self.contract = None
        self._initialize_web3()
        
        # Cache for blockchain data
        self.cache = {}
        
        logger.info(f"Blockchain connector initialized with RPC URL: {self.rpc_url}")
        logger.info(f"Contract address: {self.contract_address}")
    
    def _initialize_web3(self):
        """Initialize Web3 connection and contract instance."""
        try:
            # Thử kết nối với nhiều RPC URLs khác nhau
            urls_to_try = [
                "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94",  # Polygon Amoy testnet (từ .env)
                self.rpc_url,  # RPC URL được cung cấp khi khởi tạo
                "https://rpc-mumbai.maticvigil.com",  # Polygon Mumbai testnet
                "https://rpc-mainnet.maticvigil.com",  # Polygon Mainnet
                "http://127.0.0.1:7545",  # Ganache default
                "http://localhost:8545"   # Standard local Ethereum node
            ]
            
            for url in urls_to_try:
                try:
                    logger.info(f"Attempting to connect to blockchain at {url}")
                    self.w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 5}))
                    # Add middleware for POA chains like Polygon
                    self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                    
                    # Check connection with timeout
                    connection_success = self.w3.is_connected()
                    if connection_success:
                        logger.info(f"Successfully connected to blockchain at {url}")
                        # Cập nhật URL nếu kết nối thành công với URL khác
                        if url != self.rpc_url:
                            self.rpc_url = url
                            logger.info(f"Updated RPC URL to {url}")
                        break
                    else:
                        logger.warning(f"Could not connect to blockchain at {url}")
                except Exception as conn_error:
                    logger.warning(f"Failed to connect to blockchain at {url}: {str(conn_error)}")
            
            # Nếu không kết nối được với bất kỳ URL nào
            if not self.w3 or not self.w3.is_connected():
                logger.warning("Failed to connect to any blockchain endpoint, will use mock data")
                return False
            
            # Load contract ABI
            if self.contract_address and self.contract_abi_path:
                try:
                    if os.path.exists(self.contract_abi_path):
                        with open(self.contract_abi_path, 'r') as f:
                            contract_json = json.load(f)
                            contract_abi = contract_json.get('abi', contract_json)
                        
                        # Initialize contract
                        self.contract = self.w3.eth.contract(
                            address=self.w3.to_checksum_address(self.contract_address),
                            abi=contract_abi
                        )
                        logger.info(f"Contract initialized at {self.contract_address}")
                        return True
                    else:
                        logger.warning(f"ABI file not found at {self.contract_abi_path}, will use mock data")
                        return False
                except Exception as e:
                    logger.warning(f"Failed to initialize contract: {str(e)}, will use mock data")
                    return False
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize Web3: {str(e)}, will use mock data")
            return False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to blockchain."""
        return self.w3 is not None and self.w3.is_connected()
    
    def get_latest_block_number(self) -> int:
        """Get the latest block number."""
        if not self.is_connected():
            logger.error("Not connected to blockchain")
            return -1
        
        try:
            return self.w3.eth.block_number
        except Exception as e:
            logger.error(f"Failed to get latest block number: {str(e)}")
            return -1
    
    def get_block(self, block_number: int) -> Dict[str, Any]:
        """
        Get block data by block number.
        
        Args:
            block_number: Block number to fetch
            
        Returns:
            Block data as dictionary
        """
        if not self.is_connected():
            logger.error("Not connected to blockchain")
            return {}
        
        cache_key = f"block_{block_number}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
            # Convert to dictionary and cache
            block_dict = dict(block)
            # Convert bytes to hex strings for JSON serialization
            for key, value in block_dict.items():
                if isinstance(value, bytes):
                    block_dict[key] = value.hex()
            
            self.cache[cache_key] = block_dict
            return block_dict
        except Exception as e:
            logger.error(f"Failed to get block {block_number}: {str(e)}")
            return {}
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction data by transaction hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as dictionary
        """
        if not self.is_connected():
            logger.error("Not connected to blockchain")
            return {}
        
        cache_key = f"tx_{tx_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            tx_dict = dict(tx)
            # Convert bytes to hex strings for JSON serialization
            for key, value in tx_dict.items():
                if isinstance(value, bytes):
                    tx_dict[key] = value.hex()
            
            self.cache[cache_key] = tx_dict
            return tx_dict
        except Exception as e:
            logger.error(f"Failed to get transaction {tx_hash}: {str(e)}")
            return {}
    
    def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction receipt by transaction hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt as dictionary
        """
        if not self.is_connected():
            logger.error("Not connected to blockchain")
            return {}
        
        cache_key = f"receipt_{tx_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            receipt_dict = dict(receipt)
            # Convert bytes to hex strings for JSON serialization
            for key, value in receipt_dict.items():
                if isinstance(value, bytes):
                    receipt_dict[key] = value.hex()
            
            self.cache[cache_key] = receipt_dict
            return receipt_dict
        except Exception as e:
            logger.error(f"Failed to get transaction receipt {tx_hash}: {str(e)}")
            return {}
    
    def get_contract_events(self, 
                           event_name: str, 
                           from_block: int = 0, 
                           to_block: int = 'latest',
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get contract events by event name and block range.
        
        Args:
            event_name: Name of the event to fetch
            from_block: Starting block number
            to_block: Ending block number
            filters: Event filters
            
        Returns:
            List of events as dictionaries
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return []
        
        cache_key = f"events_{event_name}_{from_block}_{to_block}_{json.dumps(filters or {}, cls=NumpyJSONEncoder)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Get event object from contract
            event_obj = getattr(self.contract.events, event_name)
            if not event_obj:
                logger.error(f"Event {event_name} not found in contract")
                return []
            
            # Fetch events
            events = event_obj.get_logs(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters=filters
            )
            
            # Convert to dictionaries
            event_dicts = []
            for event in events:
                event_dict = dict(event)
                # Extract args
                if 'args' in event_dict:
                    event_dict.update(dict(event_dict['args']))
                # Convert bytes to hex strings for JSON serialization
                for key, value in event_dict.items():
                    if isinstance(value, bytes):
                        event_dict[key] = value.hex()
                event_dicts.append(event_dict)
            
            self.cache[cache_key] = event_dicts
            return event_dicts
        except Exception as e:
            logger.error(f"Failed to get events {event_name}: {str(e)}")
            return []
    
    def call_contract_function(self, 
                              function_name: str, 
                              *args, 
                              **kwargs) -> Any:
        """
        Call a contract function.
        
        Args:
            function_name: Name of the function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return None
        
        try:
            # Get function object from contract
            function_obj = getattr(self.contract.functions, function_name)
            if not function_obj:
                logger.error(f"Function {function_name} not found in contract")
                return None
            
            # Call function
            result = function_obj(*args).call(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Failed to call function {function_name}: {str(e)}")
            return None
    
    def get_all_nodes(self) -> List[str]:
        """
        Get all node addresses from the contract.
        
        Returns:
            List of node addresses
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # Try to get nodes from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get all nodes:
            # return self.call_contract_function('getAllNodes')
            
            # If no direct function exists, we can try to extract from events
            # For example, get all unique addresses from NodeRegistered events
            node_events = self.get_contract_events('NodeRegistered', from_block=0)
            node_addresses = set()
            for event in node_events:
                if 'nodeAddress' in event:
                    node_addresses.add(event['nodeAddress'])
            
            return list(node_addresses)
        except Exception as e:
            logger.error(f"Failed to get all nodes: {str(e)}")
            return []
    
    def get_node_reputation(self, node_address: str) -> int:
        """
        Get reputation score for a node.
        
        Args:
            node_address: Address of the node
            
        Returns:
            Reputation score
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return 0
        
        try:
            # Try to get reputation from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get node reputation:
            # return self.call_contract_function('getNodeReputation', node_address)
            
            # If no direct function exists, we can try to calculate from events
            # For example, count successful validations
            validation_events = self.get_contract_events(
                'BatchValidated', 
                from_block=0,
                filters={'validator': node_address}
            )
            
            return len(validation_events)
        except Exception as e:
            logger.error(f"Failed to get reputation for node {node_address}: {str(e)}")
            return 0
    
    def get_node_transactions(self, node_address: str) -> List[Dict[str, Any]]:
        """
        Get transactions related to a node.
        
        Args:
            node_address: Node address
            
        Returns:
            List of transactions
        """
        if not self.is_connected() or not self.contract:
            logger.warning("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # This would typically query the blockchain for transactions
            # For demo purposes, return mock transactions
            # In a real implementation, you would query for transactions involving the node
            return [
                {
                    "hash": f"0x{node_address[2:6]}tx1",
                    "timestamp": int(time.time()) - 3600,
                    "value": "0.1"
                },
                {
                    "hash": f"0x{node_address[2:6]}tx2",
                    "timestamp": int(time.time()) - 1800,
                    "value": "0.2"
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get node transactions: {str(e)}")
            return []
    
    def get_node_events(self, node_address: str) -> List[Dict[str, Any]]:
        """
        Get events related to a node.
        
        Args:
            node_address: Node address
            
        Returns:
            List of events
        """
        if not self.is_connected() or not self.contract:
            logger.warning("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # This would typically query the blockchain for events
            # For demo purposes, return mock events
            # In a real implementation, you would query for events involving the node
            return [
                {
                    "event": "BatchValidated",
                    "batchId": "1",
                    "timestamp": int(time.time()) - 3000
                },
                {
                    "event": "BatchProposed",
                    "batchId": "2",
                    "timestamp": int(time.time()) - 2000
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get node events: {str(e)}")
            return []
    
    def get_batch_events(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Get events related to a batch.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            List of events
        """
        if not self.is_connected() or not self.contract:
            logger.warning("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # This would typically query the blockchain for events
            # For demo purposes, return mock events
            # In a real implementation, you would query for events involving the batch
            return [
                {
                    "event": "BatchCreated",
                    "timestamp": int(time.time()) - 4000
                },
                {
                    "event": "BatchValidated",
                    "timestamp": int(time.time()) - 3500
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get batch events: {str(e)}")
            return []
    
    def get_dispute_events(self, dispute_id: str) -> List[Dict[str, Any]]:
        """
        Get events related to a dispute.
        
        Args:
            dispute_id: Dispute ID
            
        Returns:
            List of events
        """
        if not self.is_connected() or not self.contract:
            logger.warning("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # This would typically query the blockchain for events
            # For demo purposes, return mock events
            # In a real implementation, you would query for events involving the dispute
            return [
                {
                    "event": "DisputeCreated",
                    "timestamp": int(time.time()) - 2500
                },
                {
                    "event": "DisputeResolved",
                    "timestamp": int(time.time()) - 1500
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get dispute events: {str(e)}")
            return []
    
    def get_batch_details(self, batch_id: int) -> Dict[str, Any]:
        """
        Get details for a batch.
        
        Args:
            batch_id: ID of the batch
            
        Returns:
            Batch details as dictionary
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return {}
        
        try:
            # Try to get batch details from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get batch details:
            return self.call_contract_function('getBatchDetails', batch_id)
        except Exception as e:
            logger.error(f"Failed to get details for batch {batch_id}: {str(e)}")
            return {}
    
    def get_dispute_details(self, dispute_id: int) -> Dict[str, Any]:
        """
        Get details for a dispute.
        
        Args:
            dispute_id: ID of the dispute
            
        Returns:
            Dispute details as dictionary
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return {}
        
        try:
            # Try to get dispute details from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get dispute details:
            return self.call_contract_function('getDisputeDetails', dispute_id)
        except Exception as e:
            logger.error(f"Failed to get details for dispute {dispute_id}: {str(e)}")
            return {}
    
    def get_all_batches(self) -> List[int]:
        """
        Get all batch IDs from the contract.
        
        Returns:
            List of batch IDs
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # Try to get batches from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get all batches:
            # return self.call_contract_function('getAllBatches')
            
            # If no direct function exists, we can try to extract from events
            batch_events = self.get_contract_events('BatchProposed', from_block=0)
            batch_ids = set()
            for event in batch_events:
                if 'batchId' in event:
                    batch_ids.add(int(event['batchId']))
            
            return sorted(list(batch_ids))
        except Exception as e:
            logger.error(f"Failed to get all batches: {str(e)}")
            return []
    
    def get_all_disputes(self) -> List[int]:
        """
        Get all dispute IDs from the contract.
        
        Returns:
            List of dispute IDs
        """
        if not self.is_connected() or not self.contract:
            logger.error("Not connected to blockchain or contract not initialized")
            return []
        
        try:
            # Try to get disputes from contract
            # This is a placeholder - actual implementation depends on contract structure
            # For example, if there's a function to get all disputes:
            # return self.call_contract_function('getAllDisputes')
            
            # If no direct function exists, we can try to extract from events
            dispute_events = self.get_contract_events('DisputeCreated', from_block=0)
            dispute_ids = set()
            for event in dispute_events:
                if 'disputeId' in event:
                    dispute_ids.add(int(event['disputeId']))
            
            return sorted(list(dispute_ids))
        except Exception as e:
            logger.error(f"Failed to get all disputes: {str(e)}")
            return []
    
    def get_context_data(self) -> Dict[str, Any]:
        """
        Get context data for FL models.
        This method tries to get real-time data from blockchain, but falls back to cached mock data if needed.
        
        Returns:
            Context data as dictionary with nodes, batches, and disputes
        """
        # Initialize empty context data structure
        context_data = {
            "timestamp": datetime.now().isoformat(),
            "contractAddress": self.contract_address,
            "rpcUrl": self.rpc_url,
            "nodes": {},
            "batches": {},
            "disputes": {}
        }
        
        # Kiểm tra nếu kết nối blockchain hoạt động
        if self.w3 and self.contract and self.is_connected():
            logger.info("Getting live blockchain data...")
            try:
                # Lấy dữ liệu từ blockchain
                node_addresses = self.get_all_nodes()
                for node_address in node_addresses:
                    try:
                        reputation = self.get_node_reputation(node_address)
                        transactions = self.get_node_transactions(node_address)
                        events = self.get_node_events(node_address)
                        
                        context_data["nodes"][node_address] = {
                            "type": "node",
                            "address": node_address,
                            "reputation": reputation,
                            "transactions": transactions,
                            "events": events
                        }
                    except Exception as node_e:
                        logger.warning(f"Error processing node {node_address}: {str(node_e)}")
                
                batch_ids = self.get_all_batches()
                for batch_id in batch_ids:
                    try:
                        batch_details = self.get_batch_details(batch_id)
                        batch_events = self.get_batch_events(batch_id)
                        
                        batch_data = batch_details.copy()
                        batch_data["type"] = "batch"
                        batch_data["events"] = batch_events
                        
                        context_data["batches"][str(batch_id)] = batch_data
                    except Exception as batch_e:
                        logger.warning(f"Error processing batch {batch_id}: {str(batch_e)}")
                
                dispute_ids = self.get_all_disputes()
                for dispute_id in dispute_ids:
                    try:
                        dispute_details = self.get_dispute_details(dispute_id)
                        dispute_events = self.get_dispute_events(dispute_id)
                        
                        dispute_data = dispute_details.copy()
                        dispute_data["type"] = "dispute"
                        dispute_data["events"] = dispute_events
                        
                        context_data["disputes"][str(dispute_id)] = dispute_data
                    except Exception as dispute_e:
                        logger.warning(f"Error processing dispute {dispute_id}: {str(dispute_e)}")
                
                # Nếu có dữ liệu từ blockchain, lưu vào cache
                if (context_data["nodes"] or context_data["batches"] or context_data["disputes"]):
                    try:
                        cache_file = os.path.join(self.cache_dir, "context_data.json")
                        with open(cache_file, 'w') as f:
                            json.dump(context_data, f, indent=2, cls=NumpyJSONEncoder)
                        logger.info(f"Blockchain data cached to {cache_file}")
                        return context_data
                    except Exception as cache_e:
                        logger.error(f"Failed to save data to cache: {str(cache_e)}")
            except Exception as e:
                logger.error(f"Failed to get context data from blockchain: {str(e)}")
        
        # Nếu không thể lấy dữ liệu thực từ blockchain hoặc dữ liệu thực rỗng, dùng mock data
        logger.info("Using mock data from cache...")
        
        # Thử đọc từ cache
        cache_file = os.path.join(self.cache_dir, "context_data.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    mock_data = json.load(f)
                    logger.info(f"Loaded mock data from {cache_file}")
                    if "nodes" in mock_data and mock_data["nodes"] and isinstance(mock_data["nodes"], dict):
                        logger.info(f"Found {len(mock_data['nodes'])} nodes in mock data")
                        return mock_data
            except Exception as cache_e:
                logger.error(f"Failed to load mock data from cache: {str(cache_e)}")
        
        # Nếu không có cache, tìm kiếm file demo_context.json
        try:
            demo_paths = [
                os.path.join(self.cache_dir, "demo_context.json"),
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"
                ),
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "test_context.json"
                )
            ]
            
            for demo_path in demo_paths:
                if os.path.exists(demo_path):
                    with open(demo_path, 'r') as f:
                        mock_data = json.load(f)
                        logger.info(f"Loaded mock data from {demo_path}")
                        
                        # Lưu vào cache để sử dụng lần sau
                        try:
                            with open(cache_file, 'w') as f:
                                json.dump(mock_data, f, indent=2, cls=NumpyJSONEncoder)
                        except Exception:
                            pass
                        
                        return mock_data
        except Exception as demo_e:
            logger.error(f"Failed to load any mock data: {str(demo_e)}")
        
        # Nếu tất cả đều thất bại, trả về dữ liệu rỗng với cấu trúc hợp lệ
        return context_data
    
    def get_sybil_attack_log(self) -> Dict[str, Any]:
        """
        Get Sybil attack log data.
        This is a replacement for reading from sybil_attack_log.json.
        
        Returns:
            Sybil attack log data as dictionary
        """
        try:
            # Try to load from sybil_attack_log.json
            sybil_log_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json"
            )
            
            if os.path.exists(sybil_log_path):
                with open(sybil_log_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Sybil attack log file not found at {sybil_log_path}")
                return {}
        except Exception as e:
            logger.error(f"Failed to get Sybil attack log: {str(e)}")
            return {}
    
    def detect_attack_mode(self) -> bool:
        """
        Detect if the system is in attack mode based on blockchain data.
        
        Returns:
            True if attack mode detected, False otherwise
        """
        try:
            # Try to load Sybil attack log
            sybil_log = self.get_sybil_attack_log()
            if sybil_log and "sybilNodes" in sybil_log and len(sybil_log["sybilNodes"]) > 0:
                logger.info("Attack mode detected from Sybil attack log")
                return True
            
            # If no Sybil log, try to detect from blockchain data
            # For example, check for suspicious patterns in recent events
            
            # Get recent batch validation events
            recent_validations = self.get_contract_events('BatchValidated', from_block=-1000)
            
            # Check for unusual patterns
            # For example, if many validations in a short time from new nodes
            if len(recent_validations) > 50:  # Arbitrary threshold
                logger.info("Attack mode suspected based on high validation activity")
                return True
            
            logger.info("No attack mode detected")
            return False
        except Exception as e:
            logger.error(f"Failed to detect attack mode: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache = {}
        logger.info("Cache cleared")
    
    def save_cache(self, filename: str = "blockchain_cache.json"):
        """
        Save the cache to a file.
        
        Args:
            filename: Name of the cache file
        """
        cache_file = os.path.join(self.cache_dir, filename)
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Cache saved to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def load_cache(self, filename: str = "blockchain_cache.json"):
        """
        Load the cache from a file.
        
        Args:
            filename: Name of the cache file
        """
        cache_file = os.path.join(self.cache_dir, filename)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Cache loaded from {cache_file}")
            except Exception as e:
                logger.error(f"Failed to load cache: {str(e)}")
    
    def get_block_with_contract_activity(self, block_number: int) -> Dict[str, Any]:
        """
        Get block data with all contract-related transactions and events.
        Returns a dict with block info, contract_transactions, contract_events.
        """
        if not self.is_connected():
            logger.error("Not connected to blockchain.")
            return {}
        try:
            block = self.w3.eth.get_block(block_number, full_transactions=True)
            block_data = dict(block)
            contract_addr = self.contract_address.lower()
            contract_txs = []
            contract_events = []
            for tx in block_data.get('transactions', []):
                # 'to' can be None for contract creation
                tx_to = tx.get('to')
                if tx_to and tx_to.lower() == contract_addr:
                    contract_txs.append(dict(tx))
                    try:
                        receipt = self.w3.eth.get_transaction_receipt(tx['hash'])
                        for log in receipt['logs']:
                            if log['address'].lower() == contract_addr:
                                # Convert log to dict for JSON serializable
                                log_dict = dict(log)
                                contract_events.append(log_dict)
                    except Exception as e:
                        logger.warning(f"Failed to get receipt/logs for tx {tx['hash']}: {e}")
            block_data['contract_transactions'] = contract_txs
            block_data['contract_events'] = contract_events
            return block_data
        except Exception as e:
            logger.error(f"Failed to fetch block with contract activity: {e}")
            return {}
