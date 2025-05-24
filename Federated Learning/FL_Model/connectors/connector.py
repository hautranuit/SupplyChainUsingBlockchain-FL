import os
import json
import logging
from web3 import Web3
from web3.middleware import geth_poa_middleware # For PoA networks like Amoy

logger = logging.getLogger("blockchain_connector")

class BlockchainConnector:
    """Handles connection and data fetching from the blockchain."""

    def __init__(self, rpc_url, contract_address=None, contract_abi_path=None):
        """Initialize connection to the blockchain."""
        self.rpc_url = rpc_url
        self.contract_address = contract_address
        self.contract_abi = None
        self.web3 = None
        self.contract = None

        if not self.rpc_url:
            logger.error("RPC URL is required for BlockchainConnector.")
            raise ValueError("RPC URL cannot be empty.")

        try:
            self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # Inject middleware for PoA networks like Amoy, Rinkeby, Goerli, etc.
            self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)

            if not self.web3.is_connected():
                logger.error(f"Failed to connect to blockchain node at {self.rpc_url}")
                raise ConnectionError(f"Unable to connect to {self.rpc_url}")
            else:
                logger.info(f"Successfully connected to blockchain node at {self.rpc_url}")
                logger.info(f"Chain ID: {self.web3.eth.chain_id}, Latest Block: {self.web3.eth.block_number}")

        except Exception as e:
            logger.error(f"Error initializing Web3 connection: {str(e)}")
            raise

        if contract_address and contract_abi_path:
            try:
                # Try multiple potential paths for ABI
                potential_abi_paths = [
                    contract_abi_path,
                    os.path.join(os.path.dirname(__file__), contract_abi_path),
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), contract_abi_path),
                     # Add path relative to project root if structure is known
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SupplyChain_dapp', 'artifacts', 'contracts', 'SupplyChainNFT.sol', 'SupplyChainNFT.json'))
                ]
                
                abi_found = False
                for path in potential_abi_paths:
                     if os.path.exists(path):
                        logger.info(f"Loading ABI from: {path}")
                        with open(path, 'r') as f:
                            abi_data = json.load(f)
                            # Check if the loaded JSON contains the ABI list directly or nested
                            if isinstance(abi_data, list):
                                self.contract_abi = abi_data
                            elif isinstance(abi_data, dict) and 'abi' in abi_data:
                                self.contract_abi = abi_data['abi']
                            else:
                                logger.error(f"ABI format not recognized in {path}")
                                continue # Try next path
                        abi_found = True
                        break # Exit loop once ABI is found
                
                if not abi_found:
                    logger.error(f"Contract ABI file not found at specified paths starting with {contract_abi_path}")
                    raise FileNotFoundError(f"ABI file not found.")

                self.contract_address = Web3.to_checksum_address(self.contract_address)
                self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.contract_abi)
                logger.info(f"Contract object created for address: {self.contract_address}")

            except FileNotFoundError as e:
                 logger.error(f"ABI file error: {str(e)}")
                 # Allow connector to exist without contract for general blockchain interaction
                 self.contract = None
                 self.contract_address = None
                 self.contract_abi = None
            except Exception as e:
                logger.error(f"Error loading contract ABI or creating contract object: {str(e)}")
                # Allow connector to exist without contract
                self.contract = None
                self.contract_address = None
                self.contract_abi = None
        else:
             logger.warning("Contract address or ABI path not provided. Contract interaction disabled.")


    def get_latest_block_number(self):
        """Get the latest block number."""
        try:
            return self.web3.eth.block_number
        except Exception as e:
            logger.error(f"Error getting latest block number: {str(e)}")
            return None

    def get_events(self, event_name, from_block=0, to_block=\'latest\', filters=None):
        """Fetch specific events from the contract."""
        if not self.contract:
            logger.error("Contract not initialized. Cannot fetch events.")
            return []

        try:
            event_filter = self.contract.events[event_name].create_filter(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters=filters
            )
            events = event_filter.get_all_entries()
            logger.info(f"Fetched {len(events)} '{event_name}' events from block {from_block} to {to_block}.")
            # Convert event data to a more serializable format
            formatted_events = []
            for event in events:
                formatted_event = {
                    'event': event.event,
                    'address': event.address,
                    'transactionHash': event.transactionHash.hex(),
                    'blockNumber': event.blockNumber,
                    'logIndex': event.logIndex,
                    'args': {k: str(v) if isinstance(v, bytes) else v for k, v in event.args.items()} # Convert bytes to str
                }
                formatted_events.append(formatted_event)
            return formatted_events
        except ValueError as e:
             # Handle cases where the block range is too large or node limits are hit
            logger.error(f"ValueError fetching events '{event_name}': {str(e)}. The block range might be too large or the node might have query limits.")
            # Consider adding retry logic or suggesting smaller block ranges
            return []
        except Exception as e:
            logger.error(f"Error fetching events '{event_name}': {str(e)}")
            return []

    def get_all_contract_events(self, from_block=0, to_block=\'latest\'):
        """Fetch all known events emitted by the contract within a block range."""
        if not self.contract or not self.contract_abi:
            logger.error("Contract or ABI not initialized. Cannot fetch all events.")
            return []

        all_events_data = []
        event_signatures = {event['name']: self.web3.keccak(text=f"{event['name']}({','.join([inp['type'] for inp in event['inputs']])})").hex() 
                            for event in self.contract_abi if event['type'] == 'event'}
        
        logger.info(f"Fetching all events for contract {self.contract_address} from block {from_block} to {to_block}...")
        
        try:
            # Create a general filter for the contract address
            log_filter = self.web3.eth.filter({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': self.contract_address
            })
            
            logs = log_filter.get_all_entries()
            logger.info(f"Found {len(logs)} raw logs for the contract address.")

            # Process logs using the contract's ABI
            for log in logs:
                try:
                    # Find the event ABI entry that matches the log topics[0] (event signature)
                    event_abi = next((event for event in self.contract.events 
                                      if hasattr(event, 'signature') and log['topics'][0].hex() == event.signature()), None)
                    
                    if event_abi:
                        # Decode the log using the specific event ABI
                        event_data = event_abi().process_log(log)
                        formatted_event = {
                            'event': event_data.event,
                            'address': event_data.address,
                            'transactionHash': event_data.transactionHash.hex(),
                            'blockNumber': event_data.blockNumber,
                            'logIndex': event_data.logIndex,
                            'args': {k: str(v) if isinstance(v, bytes) else v for k, v in event_data.args.items()}
                        }
                        all_events_data.append(formatted_event)
                    else:
                         # Log as an unknown event if signature doesn't match known events
                         all_events_data.append({
                            'event': 'UnknownEvent',
                            'address': log.address,
                            'transactionHash': log.transactionHash.hex(),
                            'blockNumber': log.blockNumber,
                            'logIndex': log.logIndex,
                            'topics': [t.hex() for t in log.topics],
                            'data': log.data.hex()
                         })

                except Exception as decode_error:
                    logger.warning(f"Could not decode log at block {log.blockNumber}, tx {log.transactionHash.hex()}: {decode_error}")
                    # Append raw log info if decoding fails
                    all_events_data.append({
                        'event': 'DecodingError',
                        'address': log.address,
                        'transactionHash': log.transactionHash.hex(),
                        'blockNumber': log.blockNumber,
                        'logIndex': log.logIndex,
                        'topics': [t.hex() for t in log.topics],
                        'data': log.data.hex()
                    })
            
            logger.info(f"Successfully processed {len(all_events_data)} events.")
            return all_events_data

        except ValueError as e:
            logger.error(f"ValueError fetching all logs: {str(e)}. Block range might be too large or node limits hit.")
            return []
        except Exception as e:
            logger.error(f"Error fetching all logs for contract {self.contract_address}: {str(e)}")
            return []

    # --- Deprecated get_context_data --- 
    # This function was likely intended to fetch current state, not logs/transactions.
    # Keeping it here commented out for reference, but it's not suitable for logging.
    # def get_context_data(self):
    #     """Fetch context data (e.g., node info, product status) from the contract."""
    #     if not self.contract:
    #         logger.error("Contract not initialized. Cannot fetch context data.")
    #         return {}
    #     try:
    #         # Example: Fetching node count and details (adjust based on actual contract functions)
    #         # node_count = self.contract.functions.getNodeCount().call()
    #         # nodes = {}
    #         # for i in range(node_count):
    #         #     node_address = self.contract.functions.getNodeAddress(i).call()
    #         #     node_info = self.contract.functions.nodes(node_address).call()
    #         #     nodes[node_address] = {
    #         #         'role': node_info[0], 
    #         #         'nodeType': node_info[1],
    #         #         'reputation': node_info[2],
    #         #         'isVerified': node_info[3]
    #         #     }
            
    #         # Example: Fetching product details (adjust based on actual contract structure)
    #         # products = {}
    #         # ... logic to iterate through token IDs and get details ...

    #         logger.info("Fetching context data (placeholder - needs implementation based on contract functions)")
    #         # Replace with actual contract calls to get the state needed for FL context
    #         context = {
    #             "nodes": {},
    #             "products": {},
    #             "batches": {},
    #             "disputes": {}
    #             # Add other relevant state data
    #         }
    #         return context
    #     except Exception as e:
    #         logger.error(f"Error fetching context data: {str(e)}")
    #         return {}

