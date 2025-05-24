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
        self.max_block_range = 200  # Maximum block range for chunked queries (reduced from 500 to 200)
        self.max_retries = 3  # Maximum number of retries for failed queries
        self.max_chunks = 100  # Maximum number of chunks to process before using mock data
        self.use_mock_data = False  # Set to True to use mock data instead of real blockchain data

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
        Get events from a contract.
        Attempts to use create_filter first. If that fails, falls back to chunked eth_getLogs.
        """
        # Use mock data if flag is set
        if self.use_mock_data:
            self.logger.info(f"Using mock events for {event_name} because use_mock_data is True")
            mock_events = self.get_mock_events(event_name)
            return mock_events
            
        if not self.web3:
            self.logger.error("Web3 not initialized. Cannot get events.")
            return []
            
        if contract_name not in self.contracts:
            self.logger.error(f"Contract {contract_name} not loaded.")
            return []
            
        contract = self.contracts[contract_name]
        
        if not hasattr(contract.events, event_name):
            self.logger.error(f"Event {event_name} not found in contract {contract_name}.")
            self.logger.warning(f"Returning empty list for {event_name} as event not found and use_mock_data=False.")
            return []

        event_instance = getattr(contract.events, event_name)

        # Preserve original argument filter validation logic
        if argument_filters:
            event_abi = None
            for item in contract.abi:
                if item.get('type') == 'event' and item.get('name') == event_name:
                    event_abi = item
                    break
            
            if event_abi:
                valid_args = {input_item.get('name') for input_item in event_abi.get('inputs', [])}
                filtered_args = {arg_name: arg_value for arg_name, arg_value in argument_filters.items() if arg_name in valid_args}
                
                if len(filtered_args) < len(argument_filters):
                    removed_args = set(argument_filters.keys()) - set(filtered_args.keys())
                    self.logger.warning(f"Arguments {removed_args} not found in event '{event_name}' ABI, removing from filters.")
                
                argument_filters = filtered_args if filtered_args else None
            else: # Should not happen if hasattr check passed, but good for safety
                self.logger.warning(f"Could not find ABI for event {event_name} to validate argument_filters.")

        # MODIFICATION: Always use chunked eth_getLogs, bypass create_filter
        self.logger.info(f"Using chunked eth_getLogs directly for {event_name} (range {from_block}-{to_block}, args: {argument_filters}).")

        # Fallback: Use chunked eth_getLogs
        all_retrieved_logs: List[Any] = [] # Stores decoded log entries (AttributeDict)
        
        actual_from_block = from_block
        actual_to_block: int
        
        if isinstance(to_block, str) and to_block.lower() == 'latest':
            try:
                actual_to_block = self.web3.eth.block_number
            except Exception as e_block_num:
                self.logger.error(f"Failed to resolve 'latest' block for eth_getLogs of {event_name}: {e_block_num}. Cannot proceed.")
                return []
        elif isinstance(to_block, int):
            actual_to_block = to_block
        else:
            self.logger.error(f"Invalid to_block type ('{to_block}') for eth_getLogs of {event_name}. Cannot proceed.")
            return []

        current_chunk_from_block = actual_from_block
        event_signature_hash = ""
        try:
            # Ensure contract object is valid before accessing abi
            if not contract or not hasattr(contract, 'abi'):
                self.logger.error(f"Contract object or ABI is invalid for {contract_name}. Cannot get event signature.")
                return []
            event_signature = self.get_event_signature(contract, event_name) # Pass contract object
            event_signature_hash = self.web3.keccak(text=event_signature).hex()
            event_signature_hash = self.ensure_hex_prefix(event_signature_hash)
        except Exception as e_sig:
            self.logger.error(f"Failed to get event signature for {event_name}: {e_sig}")
            return []

        while current_chunk_from_block <= actual_to_block:
            current_chunk_to_block = min(current_chunk_from_block + self.max_block_range - 1, actual_to_block)
            self.logger.debug(f"Fetching {event_name} via eth_getLogs: chunk {current_chunk_from_block}-{current_chunk_to_block}")

            try:
                # For eth_getLogs, argument_filters need to be translated to topics.
                # This simplified fallback filters by event signature (topic0).
                # If argument_filters were provided for indexed fields, they are NOT used here to form specific topics.
                # The calling function (get_all_events_for_node) will handle further Python-side filtering if necessary.
                # For direct eth_getLogs usage, we need to construct topics if argument_filters are provided for indexed fields.
                
                topics_for_getlogs = [event_signature_hash]
                if argument_filters:
                    # Simplified topic construction: Assumes argument_filters keys are indexed event parameters.
                    # This part needs to be robust and aware of ABI to correctly format topics.
                    # For now, we'll keep it simple and primarily filter by event signature.
                    # Proper indexed filtering requires knowing which arguments are indexed and their types.
                    # Example: if 'nodeAddress' is indexed, topics_for_getlogs.append(self.web3.to_hex(self.web3.to_bytes(hexstr=node_address_checksummed))))
                    # This is a placeholder for more advanced topic building if needed.
                    # For now, we rely on the Python-side filtering in get_all_events_for_node for argument specifics.
                    pass


                filter_params_for_getlogs = {
                    'address': contract.address,
                    'topics': topics_for_getlogs,
                    'fromBlock': current_chunk_from_block,
                    'toBlock': current_chunk_to_block
                }
                
                raw_logs_chunk = self.web3.eth.get_logs(filter_params_for_getlogs)
                
                decoded_logs_in_chunk = []
                for raw_log in raw_logs_chunk:
                    try:
                        # Use the same event_instance for process_log
                        decoded_log = event_instance.process_log(raw_log)
                        decoded_logs_in_chunk.append(decoded_log)
                    except Exception as decode_error:
                        # Log specific raw_log details if possible, but be careful about log size
                        self.logger.error(f"Error decoding individual log for {event_name} in chunk {current_chunk_from_block}-{current_chunk_to_block}: {decode_error}. Log: {raw_log}")
                all_retrieved_logs.extend(decoded_logs_in_chunk)
                
                self.logger.info(f"Retrieved {len(decoded_logs_in_chunk)} decoded logs ({len(raw_logs_chunk)} raw) for {event_name} via eth_getLogs (chunk {current_chunk_from_block}-{current_chunk_to_block})")

            except Exception as getLogs_chunk_error:
                self.logger.error(f"eth_getLogs failed for {event_name} (chunk {current_chunk_from_block}-{current_chunk_to_block}): {getLogs_chunk_error}")
                # If a chunk fails, return what has been collected so far.
                # A more robust implementation might retry the chunk or skip it.
                processed_events = [self.log_to_dict(log) for log in all_retrieved_logs]
                self.logger.warning(f"Returning {len(processed_events)} partially collected {event_name} events due to error in chunk.")
                return processed_events
            
            if current_chunk_from_block > actual_to_block : # Should be covered by while condition, but as safeguard
                break
            if current_chunk_from_block == current_chunk_to_block and current_chunk_to_block == actual_to_block and self.max_block_range <=1 : # Prevent infinite loop if max_block_range is too small
                break
            current_chunk_from_block = current_chunk_to_block + 1
            
            # Only sleep if we made a request, successful or not, to avoid tight loop on errors before this point.
            time.sleep(0.75) # Small delay to be polite to the RPC node (increased from 0.5 to 0.75)

        processed_events = [self.log_to_dict(log) for log in all_retrieved_logs]
        self.logger.info(f"Successfully retrieved a total of {len(processed_events)} {event_name} events using chunked eth_getLogs (range {actual_from_block}-{actual_to_block}).")
        return processed_events
    
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
            self.logger.info(f"Using mock reputation score for node {node_address} (use_mock_data=True)")
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
            try:
                reputation = self.contract.functions.getNodeReputation(node_address).call()
                return reputation
            except Exception as call_error:
                self.logger.error(f"Error calling getNodeReputation for node {node_address}: {call_error}")
                # DO NOT Fallback to mock data if use_mock_data is False
                self.logger.warning(f"Returning 0 for reputation of {node_address} due to call error and use_mock_data=False.")
                return 0
        except Exception as e:
            self.logger.error(f"Error getting reputation for node {node_address}: {e}")
            return 0 # Default value on error
            
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
            self.logger.info(f"Using mock verification status for node {node_address} (use_mock_data=True)")
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
            try:
                is_verified = self.contract.functions.isNodeVerified(node_address).call()
                return is_verified
            except Exception as call_error:
                self.logger.error(f"Error calling isNodeVerified for node {node_address}: {call_error}")
                # DO NOT Fallback to mock data if use_mock_data is False
                self.logger.warning(f"Returning False for verification of {node_address} due to call error and use_mock_data=False.")
                return False
        except Exception as e:
            self.logger.error(f"Error checking verification for node {node_address}: {e}")
            return False # Default value on error

    def get_all_events_for_node(self, node_address: str, from_block: int = 0, to_block: Union[int, str] = 'latest') -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all events related to a specific node by trying to filter by common address fields 
        or fetching all and then filtering locally.
        
        Args:
            node_address: Ethereum address of the node (expected to be checksummed or lowercase).
            from_block: Block number to start from.
            to_block: Block number to end at.
            
        Returns:
            Dictionary mapping event names to lists of relevant events.
        """
        if not self.web3 or not self.contract_abi or not self.contract_name:
            self.logger.error("Blockchain connector not fully initialized (web3, ABI, or contract name missing). Cannot get events for node.")
            return {}

        node_address_lower = node_address.lower() # Ensure consistent comparison

        # Define all event names from the ABI that might be relevant.
        # It's better to iterate through ABI than to hardcode event names.
        event_names_from_abi = []
        for item in self.contract_abi:
            if item.get('type') == 'event' and item.get('name'):
                event_names_from_abi.append(item['name'])

        if not event_names_from_abi:
            self.logger.warning("No event definitions found in ABI. Cannot fetch events.")
            return {}

        all_node_specific_events: Dict[str, List[Dict[str, Any]]] = {}

        for event_name in event_names_from_abi:
            if self.use_mock_data:
                # If using mock data, get_mock_events should ideally return all mock events for that type,
                # and then we filter them here for the specific node.
                mock_events_for_type = self.get_mock_events(event_name)
                filtered_mock_events = []
                for event_data in mock_events_for_type:
                    # Check all string values in 'args' for the node_address_lower
                    if 'args' in event_data and isinstance(event_data['args'], dict):
                        if any(isinstance(val, str) and val.lower() == node_address_lower for val in event_data['args'].values()):
                            filtered_mock_events.append(event_data)
                if filtered_mock_events:
                    all_node_specific_events[event_name] = filtered_mock_events
                continue # Move to next event_name if using mock data

            # Logic for real data fetching
            event_abi_inputs = []
            for item in self.contract_abi:
                if item.get('type') == 'event' and item.get('name') == event_name:
                    event_abi_inputs = item.get('inputs', [])
                    break
            
            indexed_address_fields = [inp['name'] for inp in event_abi_inputs if inp.get('indexed') and inp.get('type') == 'address']
            non_indexed_address_fields = [inp['name'] for inp in event_abi_inputs if not inp.get('indexed') and inp.get('type') == 'address']

            fetched_events_for_this_type: List[Dict[str, Any]] = []

            # Try filtering by indexed address fields first (more efficient)
            if indexed_address_fields:
                for field_name in indexed_address_fields:
                    try:
                        events = self.get_events(
                            contract_name=self.contract_name,
                            event_name=event_name,
                            argument_filters={field_name: node_address}, # web3.py handles checksumming if node_address is not
                            from_block=from_block,
                            to_block=to_block
                        )
                        fetched_events_for_this_type.extend(events)
                        self.logger.debug(f"Fetched {len(events)} '{event_name}' events filtering by indexed arg '{field_name}' for node {node_address}")
                    except Exception as e:
                        self.logger.warning(f"Could not fetch '{event_name}' events by indexed arg '{field_name}' for node {node_address}: {e}. This might be okay if the node isn't involved via this field.")
            
            # If no events found via indexed filters, or if there are non-indexed address fields to check,
            # fetch all events for the type and filter locally.
            # This is less efficient but necessary for non-indexed fields or complex scenarios.
            if not fetched_events_for_this_type or non_indexed_address_fields:
                self.logger.debug(f"Fetching all '{event_name}' events and filtering locally for node {node_address} (due to no indexed results or non-indexed address fields).")
                all_events_of_this_type = self.get_events(
                    contract_name=self.contract_name,
                    event_name=event_name,
                    argument_filters=None, # Fetch all
                    from_block=from_block,
                    to_block=to_block
                )
                self.logger.debug(f"Retrieved {len(all_events_of_this_type)} total '{event_name}' events for local filtering.")

                for event_data in all_events_of_this_type:
                    # Check all string values in 'args' for the node_address_lower
                    if 'args' in event_data and isinstance(event_data['args'], dict):
                        if any(isinstance(val, str) and val.lower() == node_address_lower for val in event_data['args'].values()):
                            # Avoid duplicates if already added via indexed filter
                            # A simple check based on transactionHash and logIndex can suffice
                            is_duplicate = False
                            for existing_event in fetched_events_for_this_type:
                                if existing_event.get('transactionHash') == event_data.get('transactionHash') and \
                                   existing_event.get('logIndex') == event_data.get('logIndex'):
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                fetched_events_for_this_type.append(event_data)
            
            # Remove duplicates that might have been added if an event matched multiple indexed fields
            # or matched both an indexed field and the general fetch.
            unique_events = []
            seen_event_ids = set() # (transactionHash, logIndex)
            for event_data in fetched_events_for_this_type:
                event_id = (event_data.get('transactionHash'), event_data.get('logIndex'))
                if event_id not in seen_event_ids:
                    unique_events.append(event_data)
                    seen_event_ids.add(event_id)

            if unique_events:
                all_node_specific_events[event_name] = unique_events
                self.logger.info(f"Found {len(unique_events)} unique '{event_name}' events relevant to node {node_address}")
            else:
                self.logger.debug(f"No '{event_name}' events found relevant to node {node_address}")

        return all_node_specific_events

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
                        "selectedValidators": ["0x90F79bf6EB2c4f870365E785982E1f101E93b906", "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"]
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
                        "actor": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                        "timestamp": current_time - 86400
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
                        "actor": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "timestamp": current_time - 43200
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
                        "tokenId": "1",
                        "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "amount": "1000000000000000000",
                        "timestamp": current_time - 86400
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
                        "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "price": "1000000000000000000",
                        "oldCIDForVerification": "QmXyZ..."
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
                        "approve": True
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
                        "disputeId": "1",
                        "voter": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                        "candidate": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                        "timestamp": current_time - 86400
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
        elif event_name == "BatchCommitted":
            mock_events = [
                {
                    "args": {
                        "batchId": "1",
                        "success": True
                    },
                    "event": "BatchCommitted",
                    "blockNumber": 1400,
                    "transactionHash": "0xf678901234abcdeff678901234abcdeff678901234abcdeff678901234abcdef",
                    "logIndex": 0,
                    "timestamp": current_time - 86400
                }
            ]
            
        return mock_events
