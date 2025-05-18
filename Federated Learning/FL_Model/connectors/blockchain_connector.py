# Blockchain Connector for ChainFLIP FL Integration
# This module will handle connections to the Polygon Amoy blockchain
# and provide functions to interact with smart contracts.

import json
import os
from web3 import Web3
from dotenv import dotenv_values
from pathlib import Path
from web3.middleware import geth_poa_middleware

# --- Configuration --- 
# Get the current directory and project root
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent  # Go up one more level to reach the root

# Load environment variables from ifps_qr.env for RPC URL and Contract Address
ENV_FILE_PATH = PROJECT_ROOT / "w3storage-upload-script" / "ifps_qr.env"

config = {}
if os.path.exists(ENV_FILE_PATH):
    config = dotenv_values(ENV_FILE_PATH)
else:
    print(f"Warning: Environment file not found at {ENV_FILE_PATH}. Using default/placeholder values.")

# Default RPC URL if not found in env file (should be replaced by actual if not present)
RPC_URL = config.get("POLYGON_AMOY_RPC", "https://rpc-amoy.polygon.technology/") 

# Contract Addresses - SupplyChainNFT will be loaded dynamically
# Other contracts can be added here if they have fixed addresses and are needed
CONTRACT_ADDRESSES = {
    "SupplyChainNFT": config.get("CONTRACT_ADDRESS", None), # Loaded from .env
    # Example for other contracts if needed:
    # "Marketplace": "YOUR_MARKETPLACE_ADDRESS", 
}

# Contract ABIs: SupplyChainNFT ABI path is now relative to project root
CONTRACT_ABI_PATHS = {
    "SupplyChainNFT": PROJECT_ROOT / "SupplyChain_dapp" / "artifacts" / "contracts" / "SupplyChainNFT.sol" / "SupplyChainNFT.json",
    # "Marketplace": "/path/to/Marketplace.json",
}
CONTRACT_ABIS_CACHE = {} # Cache for loaded ABIs

def load_abi_from_file(contract_name):
    """Loads ABI for a given contract name from its specific JSON file path."""
    abi_file_path = CONTRACT_ABI_PATHS.get(contract_name)
    if not abi_file_path:
        print(f"Error: ABI file path not configured for {contract_name}.")
        return None
    
    if not os.path.exists(abi_file_path):
        print(f"Error: ABI file not found for {contract_name} at {abi_file_path}")
        return None
        
    try:
        with open(abi_file_path, 'r') as f:
            contract_json = json.load(f)
            # The ABI is usually under the "abi" key in Hardhat/Foundry artifact files
            abi = contract_json.get("abi")
            if abi:
                CONTRACT_ABIS_CACHE[contract_name] = abi # Cache it
                return abi
            else:
                print(f"Error: 'abi' key not found in JSON file for {contract_name} at {abi_file_path}")
                return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode ABI JSON for {contract_name} from {abi_file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading ABI for {contract_name}: {e}")
        return None

class BlockchainConnector:
    def __init__(self, rpc_url_override=None):
        selected_rpc_url = rpc_url_override if rpc_url_override else RPC_URL
        self.w3 = Web3(Web3.HTTPProvider(selected_rpc_url))
        
        # Thêm middleware cho POA chain
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain RPC at {selected_rpc_url}")
        print(f"Successfully connected to blockchain RPC at {selected_rpc_url}")
        self._load_contracts()

    def _load_contracts(self):
        self.contracts = {}
        for name, address in CONTRACT_ADDRESSES.items():
            if not address: # Skip if address is not configured (e.g. not found in .env)
                print(f"Warning: Address for contract {name} is not configured (e.g., missing in .env). Skipping.")
                continue
            
            abi = CONTRACT_ABIS_CACHE.get(name) # Try to get from cache
            if not abi:
                abi = load_abi_from_file(name) # Try to load from file
            
            if not abi:
                print(f"Warning: ABI for contract {name} could not be loaded. Skipping contract loading.")
                continue
            
            try:
                checksum_address = Web3.to_checksum_address(address)
                self.contracts[name] = self.w3.eth.contract(address=checksum_address, abi=abi)
                print(f"Contract {name} loaded successfully at address {checksum_address}.")
            except Exception as e:
                print(f"Error loading contract {name} at address {address}: {e}")

    def get_contract(self, name):
        contract = self.contracts.get(name)
        if not contract:
            # Attempt to load it if not already loaded (e.g. address was found later)
            self._load_contracts() # This might be redundant if init always loads, but good for safety
            contract = self.contracts.get(name)
            if not contract:
                 raise ValueError(f"Contract {name} not loaded or not found. Ensure address and ABI are correct.")
        return contract

    # --- Read-only functions (examples, expand as needed) ---
    def get_owner_of_nft(self, token_id):
        try:
            sc_nft_contract = self.get_contract("SupplyChainNFT")
            return sc_nft_contract.functions.ownerOf(token_id).call()
        except Exception as e:
            print(f"Error getting owner of NFT {token_id}: {e}")
            return None

    def get_product_history_cid(self, token_id):
        try:
            sc_nft_contract = self.get_contract("SupplyChainNFT")
            return sc_nft_contract.functions.cidMapping(token_id).call()
        except Exception as e:
            print(f"Error getting product history CID for NFT {token_id}: {e}")
            return None

    def get_node_reputation(self, node_address):
        try:
            sc_nft_contract = self.get_contract("SupplyChainNFT")
            checksum_address = Web3.to_checksum_address(node_address)
            return sc_nft_contract.functions.nodeReputation(checksum_address).call()
        except Exception as e:
            print(f"Error getting reputation for node {node_address}: {e}")
            return None
            
    def is_node_verified(self, node_address):
        try:
            sc_nft_contract = self.get_contract("SupplyChainNFT")
            checksum_address = Web3.to_checksum_address(node_address)
            return sc_nft_contract.functions.isVerified(checksum_address).call()
        except Exception as e:
            print(f"Error checking if node {node_address} is verified: {e}")
            return False # Default to false on error
            
    def get_node_last_action_timestamp(self, node_address):
        try:
            sc_nft_contract = self.get_contract("SupplyChainNFT")
            checksum_address = Web3.to_checksum_address(node_address)
            return sc_nft_contract.functions.lastActionTimestamp(checksum_address).call()
        except Exception as e:
            print(f"Error getting last action timestamp for node {node_address}: {e}")
            return 0 # Default to 0 on error

    # --- Event fetching functions ---
    def get_events(self, contract_name, event_name, from_block=0, to_block='latest', argument_filters=None):
        try:
            contract = self.get_contract(contract_name)
            event = contract.events[event_name]
            
            # Tạo bộ lọc sự kiện với các tham số phù hợp
            event_filter = event.create_filter(
                fromBlock=from_block,
                toBlock=to_block,
                argument_filters=argument_filters
            )
            return event_filter.get_all_entries()
        except Exception as e:
            print(f"Error fetching {event_name} events from {contract_name}: {e}")
            return []

# Example Usage (for testing this module independently)
if __name__ == '__main__':
    print("Testing BlockchainConnector...")
    try:
        # The connector will now try to use values from ifps_qr.env
        connector = BlockchainConnector()
        
        if "SupplyChainNFT" in connector.contracts:
            print("SupplyChainNFT contract seems to be loaded.")
            # Example: Test with a known function if contract is loaded
            # Replace with an actual node address from your deployment for a meaningful test
            test_node_address = "0x0000000000000000000000000000000000000001" 
            print(f"Attempting to get reputation for node: {test_node_address}")
            reputation = connector.get_node_reputation(test_node_address)
            if reputation is not None:
                print(f"Reputation of node {test_node_address}: {reputation}")
            else:
                print(f"Could not retrieve reputation for {test_node_address} (may not exist or error)." )
            
            is_verified = connector.is_node_verified(test_node_address)
            print(f"Is node {test_node_address} verified: {is_verified}")

        else:
            print("SupplyChainNFT contract not loaded. Check .env file and ABI path.")

    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

