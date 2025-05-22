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

# Attempt to import geth_poa_middleware, trying common locations
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    # For web3 v6.x, it's often directly under middleware
    from web3.middleware.geth_poa import geth_poa_middleware # Adjusted for web3 v6.x

from web3.exceptions import StaleBlockchain

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

        # Determine ABI path
        actual_abi_path = contract_abi_path
        if not actual_abi_path:
            # Default ABI path relative to this file's location
            # Expected: E:\\NAM3\\DO_AN_CHUYEN_NGANH\\SupplyChain_dapp\\artifacts\\contracts\\SupplyChainNFT.sol\\SupplyChainNFT.json
            connector_dir = os.path.dirname(os.path.abspath(__file__))
            fl_model_dir = os.path.dirname(connector_dir)
            federated_learning_dir = os.path.dirname(fl_model_dir)
            project_root_dir = os.path.dirname(federated_learning_dir) # This should be E:\\NAM3\\DO_AN_CHUYEN_NGANH
            actual_abi_path = os.path.join(project_root_dir, "SupplyChain_dapp", "artifacts", "contracts", "SupplyChainNFT.sol", "SupplyChainNFT.json")
            self.logger.info(f"Defaulting contract_abi_path to: {actual_abi_path}")
        self.contract_abi_path = actual_abi_path

        # Determine env_file_path
        actual_env_file_path = env_file_path
        if not actual_env_file_path:
            # Corrected default path calculation for ifps_qr.env
            # Expected: E:\\NAM3\\DO_AN_CHUYEN_NGANH\\w3storage-upload-script\\ifps_qr.env
            connector_dir = os.path.dirname(os.path.abspath(__file__))
            fl_model_dir = os.path.dirname(connector_dir)
            federated_learning_dir = os.path.dirname(fl_model_dir)
            project_root_dir = os.path.dirname(federated_learning_dir) # This should be E:\\NAM3\\DO_AN_CHUYEN_NGANH
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
            self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self.web3.is_connected(): # Corrected method name
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
            event_filter = getattr(contract.events, event_name).create_filter(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters=argument_filters or {}
            )
            
            events = event_filter.get_all_entries()
            logger.info(f"Retrieved {len(events)} {event_name} events from {contract_name}")
            return events
        except Exception as e:
            logger.error(f"Error getting events {event_name} from {contract_name}: {e}")
            
            # Fallback to manual block range if the filter fails
            try:
                logger.info(f"Attempting fallback method for event retrieval...")
                current_block = self.w3.eth.block_number
                if to_block == 'latest':
                    to_block = current_block
                    
                # Use smaller chunks to avoid timeout
                chunk_size = 1000
                all_events = []
                
                for chunk_start in range(from_block, to_block + 1, chunk_size):
                    chunk_end = min(chunk_start + chunk_size - 1, to_block)
                    logger.info(f"Fetching events from blocks {chunk_start} to {chunk_end}...")
                    
                    event_filter = getattr(contract.events, event_name).create_filter(
                        fromBlock=chunk_start,
                        toBlock=chunk_end,
                        argument_filters=argument_filters or {}
                    )
                    
                    chunk_events = event_filter.get_all_entries()
                    all_events.extend(chunk_events)
                    logger.info(f"Retrieved {len(chunk_events)} events from blocks {chunk_start}-{chunk_end}")
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(0.1)
                
                logger.info(f"Retrieved total {len(all_events)} {event_name} events from {contract_name} using chunked approach")
                return all_events
            except Exception as e2:
                logger.error(f"Fallback method also failed: {e2}")
                return []
    
    def get_events_from_block_range(self, event_name: str, from_block: Union[int, str], to_block: Union[int, str]) -> list:
        processed_events = []
        try:
            if not self.contract:
                self.logger.error("Contract not loaded. Cannot get events.")
                return []
            
            event_constructor = getattr(self.contract.events, event_name, None)
            if not event_constructor:
                self.logger.warning(f"Event '{event_name}' not found in contract ABI.")
                return []

            self.logger.info(f"Attempting to get logs for event {event_name} from block {from_block} to {to_block}")
            # Use get_logs() instead of create_filter().get_all_entries()
            logs = event_constructor.get_logs(fromBlock=from_block, toBlock=to_block)
            self.logger.info(f"Retrieved {len(logs)} logs for event {event_name}.")

            for log_entry in logs: # log_entry is an AttributeDict
                event_data = self.log_to_dict(log_entry) # event_data is now a dict

                # Fetch timestamp for each event and add it to the dict
                try:
                    block_number = log_entry.get('blockNumber')
                    if block_number is None:
                        self.logger.warning(f"Log entry for {event_name} missing 'blockNumber'. Cannot fetch timestamp. Log: {event_data}")
                        event_data['timestamp'] = None # Explicitly set if not found or handle as preferred
                    else:
                        block_info = self.web3.eth.get_block(block_number)
                        event_data["timestamp"] = block_info["timestamp"]
                    
                    processed_events.append(event_data)

                except Exception as e_block:
                    self.logger.error(f"Error fetching block info for event in block {log_entry.get('blockNumber')}: {e_block}. Appending event data with null timestamp.")
                    event_data['timestamp'] = None # Ensure timestamp field exists, even if null
                    processed_events.append(event_data) 
            
        except Exception as e:
            self.logger.error(f"Error getting events for {event_name} from block {from_block} to {to_block}: {e}")
            if hasattr(e, 'code') and e.code == -32000: # Catch generic RPC errors
                 self.logger.warning(f"RPC Error ({e.code}) while fetching logs for {event_name}: {getattr(e, 'message', str(e))}")
            # Add other specific error handling if needed (e.g., for specific exception types from get_logs)
            return []
        return processed_events
    
    def get_node_reputation(self, node_address: str) -> int:
        """
        Get reputation of a node
        
        Args:
            node_address: Address of the node
            
        Returns:
            Reputation value or None if error
        """
        try: # Added try-except for to_checksum_address
            checksum_address = self.web3.to_checksum_address(node_address)
        except Exception as e:
            self.logger.error(f"Invalid node address format {node_address} for reputation check: {e}")
            return 0

        if not self.contract: # Check self.contract instead of self.contracts
            self.logger.error("Contract not loaded. Cannot get node reputation.")
            return 0
        
        # ... (rest of the method remains similar, using self.contract)
        try:
            if "getNodeReputation" not in self.contract.functions:
                self.logger.warning(f"Function \'getNodeReputation\' not found in contract ABI. Returning default reputation 0 for node {checksum_address}.")
                return 0
            
            reputation = self.contract.functions.getNodeReputation(checksum_address).call()
            self.logger.info(f"Reputation for node {checksum_address}: {reputation}")
            return reputation
        except Exception as e:
            self.logger.error(f"Error getting reputation for node {checksum_address}: {e}")
            return 0

    def is_node_verified(self, node_address: str) -> bool:
        """
        Check if a node is verified
        
        Args:
            node_address: Address of the node
            
        Returns:
            True if verified, False otherwise
        """
        try: # Added try-except for to_checksum_address
            checksum_address = self.web3.to_checksum_address(node_address)
        except Exception as e:
            self.logger.error(f"Invalid node address format {node_address} for verification check: {e}")
            return False

        if not self.contract: # Check self.contract
            self.logger.error("Contract not loaded. Cannot check node verification.")
            return False

        # ... (rest of the method remains similar, using self.contract)
        try:
            if "isNodeVerified" not in self.contract.functions:
                self.logger.warning(f"Function \'isNodeVerified\' not found in contract ABI. Returning default verification status False for node {checksum_address}.")
                return False
            
            verified_status = self.contract.functions.isNodeVerified(checksum_address).call()
            self.logger.info(f"Node {checksum_address} verification status: {verified_status}")
            return verified_status
        except Exception as e:
            self.logger.error(f"Error checking if node {checksum_address} is verified: {e}")
            return False
    
    def get_node_type(self, node_address: str) -> Optional[int]:
        """
        Get type of a node (if available in contract)
        
        Args:
            node_address: Address of the node
            
        Returns:
            Node type value or None if error or not available
        """
        try: # Added try-except for to_checksum_address
            checksum_address = self.web3.to_checksum_address(node_address)
        except Exception as e:
            self.logger.error(f"Invalid node address format {node_address} for type check: {e}")
            return None

        if not self.contract: # Check self.contract
            self.logger.error("Contract not loaded. Cannot get node type.")
            return None
            
        # ... (rest of the method remains similar, using self.contract)
        try:
            if not hasattr(self.contract.functions, 'getNodeType'): # Check self.contract
                self.logger.warning("getNodeType function not available in contract")
                return None
                
            node_type = self.contract.functions.getNodeType(checksum_address).call()
            return node_type
        except Exception as e:
            self.logger.error(f"Error getting type for node {checksum_address}: {e}")
            return None
