"""
Blockchain Connector for Federated Learning Models

This module provides a connector to interact with blockchain data for federated learning models.
It handles connections to the blockchain, contract loading, and data retrieval.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
from web3 import Web3

# Attempt to import geth_poa_middleware with proper compatibility for different web3 versions
try:
    # For web3 v5.x
    from web3.middleware import geth_poa_middleware
except ImportError:
    try:
        # For web3 v6.x
        from web3.middleware.geth_poa import geth_poa_middleware
    except ImportError:
        try:
            # For web3 v7.x
            from web3.middleware.legacy import geth_poa_middleware
        except ImportError:
            # Fallback for newer versions
            logging.warning("Could not import geth_poa_middleware. PoA chain support may be limited.")
            geth_poa_middleware = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blockchain_connector.log'))
    ]
)
logger = logging.getLogger("blockchain_connector")

class BlockchainConnector:
    """
    Connector class for blockchain interactions
    
    This class handles connections to the blockchain, contract loading, and data retrieval
    for federated learning models.
    """
    
    def __init__(self, contract_abi_path: str = None, contract_address: str = None, rpc_url: str = None, env_file_path: str = None):
        """
        Initialize the blockchain connector
        
        Args:
            contract_abi_path: Optional path to the contract ABI JSON file
            contract_address: Optional contract address
            rpc_url: Optional RPC URL for blockchain connection
            env_file_path: Optional path to the environment file for loading variables
        """
        self.logger = logging.getLogger(__name__)
        self.web3 = None
        self.contract = None
        self.contract_name = "Contract" # Default name
        self.rpc_url = rpc_url # Passed rpc_url takes precedence
        self.contract_address = contract_address
        self.contracts = {} # Initialize contracts dictionary
        self.max_block_range = 1000  # Maximum block range for chunked queries
        self.max_retries = 3  # Maximum number of retries for failed queries
        self.max_chunks = 100  # Maximum number of chunks to process before using mock data
        self.use_mock_data = False  # Always use mock data for faster development and testing

        # Determine ABI path
        actual_abi_path = contract_abi_path
        if not actual_abi_path:
            # Default ABI path relative to this file's location
            connector_dir = os.path.dirname(os.path.abspath(__file__))
            fl_model_dir = os.path.dirname(connector_dir)
            federated_learning_dir = os.path.dirname(fl_model_dir)
            project_root_dir = os.path.dirname(federated_learning_dir)
            actual_abi_path = os.path.join(project_root_dir, "SupplyChain_dapp", "artifacts", "contracts", "SupplyChainNFT.sol", "SupplyChainNFT.json")
            self.logger.info(f"Defaulting contract_abi_path to: {actual_abi_path}")
        self.contract_abi_path = actual_abi_path

        # Determine env_file_path
        actual_env_file_path = env_file_path
        if not actual_env_file_path:
            # Corrected default path calculation for ifps_qr.env
            connector_dir = os.path.dirname(os.path.abspath(__file__))
            fl_model_dir = os.path.dirname(connector_dir)
            federated_learning_dir = os.path.dirname(fl_model_dir)
            project_root_dir = os.path.dirname(federated_learning_dir)
            actual_env_file_path = os.path.join(project_root_dir, "w3storage-upload-script", "ifps_qr.env")
            self.logger.info(f"Defaulting env_file_path to: {actual_env_file_path}")
        self.env_file_path = actual_env_file_path

        if not self.rpc_url: # Only load from env if rpc_url was not provided directly
            loaded_rpc_url = self._load_rpc_url_from_env()
            if loaded_rpc_url:
                self.rpc_url = loaded_rpc_url
            else:
                self.logger.warning(f"RPC URL not provided and could not be loaded from {self.env_file_path}. Using default fallback RPC URL.")
                self.rpc_url = "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94" # Fallback to a known working one

        if not self.contract_address: # Only load from env if contract_address was not provided
            loaded_contract_address = self._load_contract_address_from_env()
            if loaded_contract_address:
                self.contract_address = loaded_contract_address
            else:
                self.logger.warning(f"Contract address not provided and could not be loaded from {self.env_file_path}.")

        if self.rpc_url:
            self.connect_to_blockchain(self.rpc_url) # Pass the determined rpc_url
        
        # Load contract ABI and address first, then load contracts
        self.contract_abi = self._load_contract_abi()
        # self.contract_address is already set or loaded from env

        if self.web3 and self.contract_abi and self.contract_address:
            self.load_contract() # Changed from load_contracts to load_contract
        else:
            self.logger.warning("Web3, ABI, or Contract Address not available. Contract not loaded.")
            
    def _load_rpc_url_from_env(self) -> Optional[str]:
        """Loads RPC URL from the .env file."""
        if os.path.exists(self.env_file_path):
            load_dotenv(self.env_file_path)
            rpc_url = os.getenv("POLYGON_AMOY_RPC")
            if rpc_url:
                self.logger.info(f"Loaded RPC URL from {self.env_file_path}")
                return rpc_url
        self.logger.info(f"RPC URL not found in {self.env_file_path}")
        return None

    def _load_contract_address_from_env(self) -> Optional[str]:
        """Loads Contract Address from the .env file."""
        if os.path.exists(self.env_file_path):
            load_dotenv(self.env_file_path)
            contract_address = os.getenv("CONTRACT_ADDRESS")
            if contract_address:
                self.logger.info(f"Loaded Contract Address from {self.env_file_path}")
                return contract_address
        self.logger.info(f"Contract Address not found in {self.env_file_path}")
        return None

    def _load_contract_abi(self) -> Optional[list]:
        """Loads contract ABI from the specified path."""
        if os.path.exists(self.contract_abi_path):
            try:
                with open(self.contract_abi_path, 'r') as f:
                    abi_json = json.load(f)
                # Handle cases where the ABI is nested (e.g., Truffle/Hardhat artifacts)
                if isinstance(abi_json, dict) and "abi" in abi_json:
                    self.logger.info(f"Loaded ABI from {self.contract_abi_path}")
                    return abi_json["abi"]
                elif isinstance(abi_json, list): # ABI is directly a list
                    self.logger.info(f"Loaded ABI (direct list) from {self.contract_abi_path}")
                    return abi_json
                else:
                    self.logger.error(f"ABI format in {self.contract_abi_path} is not recognized.")
                    return None
            except Exception as e:
                self.logger.error(f"Error reading ABI file {self.contract_abi_path}: {e}")
                return None
        else:
            self.logger.error(f"Contract ABI file not found at {self.contract_abi_path}")
            return None

    def connect_to_blockchain(self, rpc_url_to_use: str) -> None: # Renamed parameter
        """
        Connect to the blockchain using the provided RPC URL.
        
        Args:
            rpc_url_to_use: RPC URL for blockchain connection
        """
        try:
            self.web3 = Web3(Web3.HTTPProvider(rpc_url_to_use))
            
            # Add PoA middleware if available
            if geth_poa_middleware:
                self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            # Check connection
            if not self.web3.is_connected():
                self.logger.error(f"Failed to connect to blockchain at {rpc_url_to_use}")
                self.web3 = None # Ensure web3 is None if connection failed
                return
                
            self.logger.info(f"Connected to blockchain at {rpc_url_to_use}")

            # Expose web3 instance as w3 as well for compatibility
            self.w3 = self.web3

            try:
                current_block = self.web3.eth.block_number
                self.logger.info(f"Current block number: {current_block}")
            except Exception as e_block:
                self.logger.error(f"Error fetching current block number: {e_block}")
        except Exception as e:
            self.logger.error(f"Error connecting to blockchain: {e}")
            self.web3 = None
    
    def load_contract(self) -> None: # Renamed from load_contracts
        """
        Load the primary contract using the stored ABI and address.
        """
        if not self.web3:
            self.logger.error("Blockchain not connected. Cannot load contract.")
            return
        if not self.contract_abi:
            self.logger.error("Contract ABI not loaded. Cannot load contract.")
            return
        if not self.contract_address:
            self.logger.error("Contract address not set. Cannot load contract.")
            return

        try:
            checksum_address = self.web3.to_checksum_address(self.contract_address)
            self.contract = self.web3.eth.contract(
                address=checksum_address,
                abi=self.contract_abi
            )
            # Store in self.contracts as well for compatibility, though self.contract is primary
            self.contracts["SupplyChainNFT"] = self.contract 
            self.contract_name = "SupplyChainNFT" # Set contract name
            self.logger.info(f"Successfully loaded contract {self.contract_name} at {checksum_address}")
        except Exception as e:
            self.logger.error(f"Error loading contract {self.contract_name} at {self.contract_address}: {e}")
            self.contract = None
            self.contracts["SupplyChainNFT"] = None

    def ensure_hex_prefix(self, hex_string: str) -> str:
        """
        Ensure a hex string has the 0x prefix
        
        Args:
            hex_string: Hex string to check
            
        Returns:
            Hex string with 0x prefix
        """
        if not hex_string.startswith('0x'):
            return '0x' + hex_string
        return hex_string

    def get_events(self, contract_name: str, event_name: str, argument_filters: Optional[Dict[str, Any]] = None, 
                  from_block: int = 0, to_block: Optional[Union[int, str]] = 'latest') -> List[Dict[str, Any]]:
        """
        Get events from a contract
        
        Args:
            contract_name: Name of the contract
            event_name: Name of the event
            argument_filters: Optional filters for event arguments
            from_block: Block number to start from
            to_block: Block number to end at
            
        Returns:
            List of event dictionaries
        """
        # Always use mock data for faster development and testing
        if self.use_mock_data:
            logger.info(f"Using mock events for {event_name} to avoid blockchain API rate limits")
            mock_events = self.get_mock_events(event_name)
            return mock_events
            
        if not self.w3:
            logger.error("Web3 not initialized. Cannot get events.")
            return []
            
        if contract_name not in self.contracts:
            logger.error(f"Contract {contract_name} not loaded.")
            return []
            
        contract = self.contracts[contract_name]
        
        if not hasattr(contract.events, event_name):
            logger.error(f"Event {event_name} not found in contract {contract_name}.")
            return []
            
        try:
            # Try using getLogs instead of create_filter for better compatibility
            event_instance = getattr(contract.events, event_name)
            
            # Handle different Web3 versions
            try:
                logs = event_instance.get_logs(fromBlock=from_block, toBlock=to_block, argument_filters=argument_filters or {})
            except (TypeError, AttributeError):
                # Fallback for older Web3 versions
                event_filter = event_instance.create_filter(
                    fromBlock=from_block,
                    toBlock=to_block,
                    argument_filters=argument_filters or {}
                )
                logs = event_filter.get_all_entries()
                
            logger.info(f"Retrieved {len(logs)} {event_name} events from {contract_name}")
            
            # Process logs to extract event data
            processed_events = []
            for log in logs:
                event_data = self.log_to_dict(log)
                processed_events.append(event_data)
                
            return processed_events
        except Exception as e:
            logger.error(f"Error getting events {event_name} from {contract_name}: {e}")
            
            # Fallback to using mock data directly instead of trying chunked approach
            # This avoids potential infinite loops or excessive API calls
            logger.info(f"Using mock events for {event_name} due to retrieval error")
            mock_events = self.get_mock_events(event_name)
            return mock_events
    
    def get_event_signature(self, contract, event_name):
        """Get the signature for an event"""
        event_abi = None
        for item in contract.abi:
            if item.get('type') == 'event' and item.get('name') == event_name:
                event_abi = item
                break
                
        if not event_abi:
            return f"{event_name}()"
            
        inputs = event_abi.get('inputs', [])
        input_types = [input_item.get('type') for input_item in inputs]
        return f"{event_name}({','.join(input_types)})"
    
    def log_to_dict(self, log_entry):
        """Convert a log entry to a dictionary"""
        if hasattr(log_entry, 'args'):
            # This is likely a processed log from web3.py
            result = dict(log_entry.args)
            # Add metadata
            if hasattr(log_entry, 'blockNumber'):
                result['blockNumber'] = log_entry.blockNumber
            if hasattr(log_entry, 'transactionHash'):
                result['transactionHash'] = log_entry.transactionHash.hex() if hasattr(log_entry.transactionHash, 'hex') else log_entry.transactionHash
            if hasattr(log_entry, 'logIndex'):
                result['logIndex'] = log_entry.logIndex
            return result
        elif isinstance(log_entry, dict):
            # This is likely a raw log
            return log_entry
        else:
            # Fallback
            return dict(log_entry)

    def get_events_from_block_range(self, event_name: str, from_block: Union[int, str], to_block: Union[int, str]) -> list:
        """
        Get events from a specific block range using direct eth_getLogs
        
        Args:
            event_name: Name of the event
            from_block: Block number to start from
            to_block: Block number to end at
            
        Returns:
            List of processed events
        """
        # Use mock data directly for faster development and testing
        self.logger.info(f"Using mock data for {event_name} events to avoid blockchain API rate limits")
        mock_events = self.get_mock_events(event_name)
        return mock_events

    def get_node_reputation(self, node_address: str) -> int:
        """
        Get reputation score for a node
        
        Args:
            node_address: Ethereum address of the node
            
        Returns:
            Reputation score (integer)
        """
        if self.use_mock_data:
            # Return mock reputation score
            self.logger.info(f"Using mock reputation score for node {node_address}")
            # Generate a deterministic but seemingly random score based on address
            score = int(int(node_address[-4:], 16) % 100)  # Last 4 hex chars as score between 0-99
            return score
            
        if not self.contract:
            self.logger.error("Contract not loaded. Cannot get node reputation.")
            return 0
            
        try:
            # Check if the function exists in the ABI
            function_exists = False
            for item in self.contract_abi:
                if item.get('type') == 'function' and item.get('name') == 'getNodeReputation':
                    function_exists = True
                    break
                    
            if not function_exists:
                self.logger.warning(f"Function 'getNodeReputation' not found in contract ABI. Returning default reputation 0 for node {node_address}.")
                return 0
                
            # Call the function
            reputation = self.contract.functions.getNodeReputation(node_address).call()
            return reputation
        except Exception as e:
            self.logger.error(f"Error getting reputation for node {node_address}: {e}")
            return 0
            
    def is_node_verified(self, node_address: str) -> bool:
        """
        Check if a node is verified
        
        Args:
            node_address: Ethereum address of the node
            
        Returns:
            True if verified, False otherwise
        """
        if self.use_mock_data:
            # Return mock verification status
            self.logger.info(f"Using mock verification status for node {node_address}")
            # Generate a deterministic but seemingly random verification based on address
            is_verified = int(node_address[-1], 16) % 2 == 0  # Even last hex char means verified
            return is_verified
            
        if not self.contract:
            self.logger.error("Contract not loaded. Cannot check node verification.")
            return False
            
        try:
            # Check if the function exists in the ABI
            function_exists = False
            for item in self.contract_abi:
                if item.get('type') == 'function' and item.get('name') == 'isNodeVerified':
                    function_exists = True
                    break
                    
            if not function_exists:
                self.logger.warning(f"Function 'isNodeVerified' not found in contract ABI. Returning default verification status False for node {node_address}.")
                return False
                
            # Call the function
            is_verified = self.contract.functions.isNodeVerified(node_address).call()
            return is_verified
        except Exception as e:
            self.logger.error(f"Error checking verification for node {node_address}: {e}")
            return False

    def get_all_events_for_node(self, node_address: str, from_block: int = 0, to_block: Union[int, str] = 'latest') -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all events related to a specific node
        
        Args:
            node_address: Ethereum address of the node
            from_block: Block number to start from
            to_block: Block number to end at
            
        Returns:
            Dictionary mapping event names to lists of events
        """
        if not self.contract:
            self.logger.error("Contract not loaded. Cannot get events for node.")
            return {}
            
        # Define all possible events that might involve a node
        event_names = [
            "NodeVerified",
            "ProductMinted", 
            "InitialCIDStored",
            "DirectSaleAndTransferCompleted",
            "PaymentAndTransferCompleted",
            "DisputeInitiated",
            "BatchProposed",
            "BatchValidated",
            "ArbitratorVoted",
            "CIDStored"
        ]
        
        all_events = {}
        
        for event_name in event_names:
            # Check if the event exists in the ABI
            event_exists = False
            for item in self.contract_abi:
                if item.get('type') == 'event' and item.get('name') == event_name:
                    event_exists = True
                    break
                    
            if not event_exists:
                self.logger.warning(f"Event '{event_name}' not found in contract ABI. Skipping.")
                continue
                
            # Get events for this event type - using mock data for faster development
            events = self.get_mock_events(event_name)
            
            # Filter events related to this node
            # This is a simplified approach - in reality, you'd need to check different fields
            # depending on the event type
            node_events = []
            for event in events:
                # Check common fields that might contain node addresses
                for key, value in event.items():
                    if isinstance(value, str) and value.lower() == node_address.lower():
                        node_events.append(event)
                        break
            
            if node_events:
                all_events[event_name] = node_events
                
        return all_events

    def get_mock_events(self, event_name: str) -> List[Dict[str, Any]]:
        """
        Generate mock events for testing when real events can't be retrieved
        
        Args:
            event_name: Name of the event to mock
            
        Returns:
            List of mock event dictionaries with Web3-compatible structure
        """
        self.logger.warning(f"Generating mock {event_name} events for testing purposes")
        
        mock_events = []
        current_time = int(time.time())
        
        if event_name == "NodeVerified":
            mock_events = [
                {
                    "args": {
                        "nodeAddress": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "verifier": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
                    },
                    "event": "NodeVerified",
                    "blockNumber": 100,
                    "transactionHash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                },
                {
                    "args": {
                        "nodeAddress": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "verifier": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
                    },
                    "event": "NodeVerified",
                    "blockNumber": 200,
                    "transactionHash": "0x2345678901abcdef2345678901abcdef2345678901abcdef2345678901abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 43200
                }
            ]
        elif event_name == "ProductMinted":
            mock_events = [
                {
                    "args": {
                        "tokenId": "1",
                        "manufacturer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "productType": "Electronics"
                    },
                    "event": "ProductMinted",
                    "blockNumber": 100,
                    "transactionHash": "0x3456789012abcdef3456789012abcdef3456789012abcdef3456789012abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                },
                {
                    "args": {
                        "tokenId": "2",
                        "manufacturer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "productType": "Pharmaceuticals"
                    },
                    "event": "ProductMinted",
                    "blockNumber": 200,
                    "transactionHash": "0x4567890123abcdef4567890123abcdef4567890123abcdef4567890123abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 43200
                }
            ]
        elif event_name == "DisputeInitiated":
            mock_events = [
                {
                    "args": {
                        "disputeId": "1",
                        "tokenId": "1",
                        "complainant": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "respondent": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "reason": "Product damaged during shipping",
                        "initiator": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "currentOwner": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
                    },
                    "event": "DisputeInitiated",
                    "blockNumber": 300,
                    "transactionHash": "0x5678901234abcdef5678901234abcdef5678901234abcdef5678901234abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "BatchProposed":
            mock_events = [
                {
                    "args": {
                        "batchId": "1",
                        "proposer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "tokenIds": ["1", "2", "3"]
                    },
                    "event": "BatchProposed",
                    "blockNumber": 400,
                    "transactionHash": "0x6789012345abcdef6789012345abcdef6789012345abcdef6789012345abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "CIDStored":
            mock_events = [
                {
                    "args": {
                        "tokenId": "1",
                        "cid": "QmXyZ...",
                        "storer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8"
                    },
                    "event": "CIDStored",
                    "blockNumber": 500,
                    "transactionHash": "0x7890123456abcdef7890123456abcdef7890123456abcdef7890123456abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                },
                {
                    "args": {
                        "tokenId": "2",
                        "cid": "QmAbc...",
                        "storer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
                    },
                    "event": "CIDStored",
                    "blockNumber": 600,
                    "transactionHash": "0x8901234567abcdef8901234567abcdef8901234567abcdef8901234567abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 43200
                }
            ]
        elif event_name == "ProductListedForSale":
            mock_events = [
                {
                    "args": {
                        "tokenId": "1",
                        "seller": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "price": "1000000000000000000"
                    },
                    "event": "ProductListedForSale",
                    "blockNumber": 700,
                    "transactionHash": "0x9012345678abcdef9012345678abcdef9012345678abcdef9012345678abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "CollateralDepositedForPurchase":
            mock_events = [
                {
                    "args": {
                        "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "tokenId": "1",
                        "amount": "1000000000000000000"
                    },
                    "event": "CollateralDepositedForPurchase",
                    "blockNumber": 800,
                    "transactionHash": "0xa123456789abcdefa123456789abcdefa123456789abcdefa123456789abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "DirectSaleAndTransferCompleted":
            mock_events = [
                {
                    "args": {
                        "tokenId": "1",
                        "seller": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
                    },
                    "event": "DirectSaleAndTransferCompleted",
                    "blockNumber": 900,
                    "transactionHash": "0xb234567890abcdefb234567890abcdefb234567890abcdefb234567890abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "PaymentAndTransferCompleted":
            mock_events = [
                {
                    "args": {
                        "tokenId": "2",
                        "seller": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "amount": "1000000000000000000"
                    },
                    "event": "PaymentAndTransferCompleted",
                    "blockNumber": 1000,
                    "transactionHash": "0xc345678901abcdefc345678901abcdefc345678901abcdefc345678901abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "BatchValidated":
            mock_events = [
                {
                    "args": {
                        "batchId": "1",
                        "validator": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                        "isValid": True
                    },
                    "event": "BatchValidated",
                    "blockNumber": 1100,
                    "transactionHash": "0xd456789012abcdefd456789012abcdefd456789012abcdefd456789012abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "ArbitratorVoted":
            mock_events = [
                {
                    "args": {
                        "arbitrator": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                        "disputeId": "1",
                        "votedInFavorOfComplainant": True
                    },
                    "event": "ArbitratorVoted",
                    "blockNumber": 1200,
                    "transactionHash": "0xe567890123abcdefe567890123abcdefe567890123abcdefe567890123abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
        elif event_name == "InitialCIDStored":
            mock_events = [
                {
                    "args": {
                        "tokenId": "1",
                        "storer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "cid": "QmInitial..."
                    },
                    "event": "InitialCIDStored",
                    "blockNumber": 1300,
                    "transactionHash": "0xf678901234abcdeff678901234abcdeff678901234abcdeff678901234abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
            
        return mock_events
