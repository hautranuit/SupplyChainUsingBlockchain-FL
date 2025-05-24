import json
import os
import logging

# Assuming BlockchainConnector is in the same directory or PYTHONPATH is set
# If not, adjust the import path accordingly.
# from FL_Model.connectors.blockchain_connector import BlockchainConnector
# For simplicity, if running this script from within the 'connectors' directory:
from blockchain_connector import BlockchainConnector

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_amoy_node_events():
    logger.info("Starting Amoy testnet event retrieval test...")

    # --- Configuration ---
    # Path to the ABI JSON file (confirmed in previous steps)
    abi_path = r"e:\NAM3\DO_AN_CHUYEN_NGANH\SupplyChain_dapp\artifacts\contracts\SupplyChainNFT.sol\SupplyChainNFT.json"
    
    # RPC URL for Polygon Amoy Testnet
    # The BlockchainConnector has a fallback, but we can be explicit.
    # If you have a specific RPC URL in an .env file that BlockchainConnector loads,
    # you might not need to pass it here if the connector is configured to find it.
    amoy_rpc_url = "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94" 
    # Alternatively, let the connector use its default or .env loaded one:
    # amoy_rpc_url = None 

    # Sample node address from demo_context.json (manufacturerAddress)
    sample_node_address = "0x04351e7dF40d04B5E610c4aA033faCf435b98711"

    # Contract address from demo_context.json
    # This is needed by BlockchainConnector
    contract_address = "0x88b7eb798735ce65eb282df165e01e30de8786e3" # From demo_context.json

    logger.info(f"Using ABI path: {abi_path}")
    logger.info(f"Using Contract Address: {contract_address}")
    if amoy_rpc_url:
        logger.info(f"Explicitly using Amoy RPC URL: {amoy_rpc_url}")
    else:
        logger.info("Relying on BlockchainConnector's default/env RPC URL for Amoy.")
    logger.info(f"Test Node Address: {sample_node_address}")

    # --- Instantiate BlockchainConnector ---
    try:
        connector = BlockchainConnector(
            contract_abi_path=abi_path,
            contract_address=contract_address, # Provide the contract address
            rpc_url=amoy_rpc_url # Pass None to use connector's default/env logic
        )
        # Ensure we are not using mock data
        connector.use_mock_data = False
        logger.info(f"BlockchainConnector instantiated. use_mock_data: {connector.use_mock_data}")

        if not connector.web3 or not connector.web3.is_connected():
            logger.error("Failed to connect to the blockchain. Please check RPC URL and network.")
            return
        
        latest_block_number = connector.web3.eth.block_number
        logger.info(f"Latest block number: {latest_block_number}")

        if not connector.contract:
            logger.error(f"Failed to load contract at address {connector.contract_address}. Check ABI and contract address.")
            return
            
        logger.info(f"Successfully connected to: {connector.rpc_url}")
        logger.info(f"Contract '{connector.contract_name}' loaded at address: {connector.contract.address}")

    except Exception as e:
        logger.error(f"Error instantiating BlockchainConnector: {e}", exc_info=True)
        return

    # --- Call get_all_events_for_node ---
    try:
        logger.info(f"Calling get_all_events_for_node for address: {sample_node_address}")
        # Define a reasonable block range if needed, or use defaults
        # from_block = 0 
        # to_block = 'latest' # Default

        # Fetch events from a recent range to avoid overwhelming the RPC node
        to_block = latest_block_number
        from_block = max(0, latest_block_number - 1000) # Look back 1000 blocks, or from block 0 if less than 1000 blocks exist
        logger.info(f"Fetching events from block {from_block} to {to_block}")

        node_events = connector.get_all_events_for_node(
            node_address=sample_node_address,
            from_block=from_block,
            to_block=to_block
        )

        logger.info(f"Events retrieved for node {sample_node_address}:")
        if node_events:
            for event_name, events_list in node_events.items():
                logger.info(f"  Event: {event_name} ({len(events_list)} occurrences)")
                for i, event_data in enumerate(events_list):
                    # Log only a summary of each event to avoid excessive output
                    event_args_summary = {k: (str(v)[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) 
                                          for k, v in event_data.get('args', {}).items()}
                    logger.info(f"    {i+1}. Block: {event_data.get('blockNumber')}, TxHash: {event_data.get('transactionHash')}, Args: {event_args_summary}")
        else:
            logger.info("  No events found for this node in the specified range.")

    except Exception as e:
        logger.error(f"Error calling get_all_events_for_node: {e}", exc_info=True)

if __name__ == "__main__":
    # Create a dummy .env file if BlockchainConnector expects one and it's not present
    # This is just for testing convenience if the connector strictly requires .env loading for some paths
    env_file_path_for_connector = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'w3storage-upload-script', 'ifps_qr.env')
    if not os.path.exists(env_file_path_for_connector):
        logger.warning(f"Potential .env file for connector not found at {env_file_path_for_connector}. Creating a dummy one if needed by connector logic.")
        # os.makedirs(os.path.dirname(env_file_path_for_connector), exist_ok=True)
        # with open(env_file_path_for_connector, 'w') as f:
        #     f.write("# Dummy .env file for testing BlockchainConnector\n")
        #     f.write(f"POLYGON_AMOY_RPC={amoy_rpc_url_for_env_if_needed}\n") # if you want to test .env loading
        #     f.write(f"CONTRACT_ADDRESS={contract_address_for_env_if_needed}\n")


    # The BlockchainConnector itself tries to find the ABI relative to its own path if not provided.
    # And it tries to load RPC/Contract address from a .env file.
    # The instantiation above is explicit for clarity in this test script.
    
    test_amoy_node_events()
    logger.info("Test script finished.")
