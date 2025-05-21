#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Add FL Model directory to path - adjusted for new directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_Model"))
from tff_sybil_detection.real_data_preparation_sybil import make_federated_data_sybil_real

def load_env_variables():
    # Try to load from w3storage-upload-script/ifps_qr.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "w3storage-upload-script", "ifps_qr.env")
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: Environment file not found at {env_path}. Using default/placeholder values.")

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def main():
    log_message("=== Starting Sybil Detection FL Model ===")
    
    # Load environment variables
    load_env_variables()
    
    # Read demo_context.json to get node addresses
    try:
        context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json")
        log_message(f"Looking for context file at: {context_path}")
        
        if not os.path.exists(context_path):
            log_message(f"Context file not found at {context_path}")
            # Try alternative path
            context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "demo_context.json")
            log_message(f"Trying alternative path: {context_path}")
            
            if not os.path.exists(context_path):
                log_message("Context file not found at alternative path either. demo_context.json is optional if sybil_attack_log.json is present.")
                context = {} # Initialize context as empty if not found, can still proceed if sybil log exists
            else:
                with open(context_path, "r", encoding='utf-8') as f:
                    context = json.load(f)
                    log_message(f"Successfully loaded context with keys: {list(context.keys())}")
        else:
            with open(context_path, "r", encoding='utf-8') as f:
                context = json.load(f)
                log_message(f"Successfully loaded context with keys: {list(context.keys())}")
    except Exception as e:
        log_message(f"Error reading demo context: {e}. Continuing without it if sybil_attack_log.json is available.")
        context = {}

    # Read sybil_attack_log.json
    sybil_log_data = None
    try:
        sybil_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json")
        log_message(f"Looking for Sybil attack log file at: {sybil_log_path}")
        if os.path.exists(sybil_log_path):
            with open(sybil_log_path, "r", encoding='utf-8') as f:
                sybil_log_data = json.load(f)
            log_message(f"Successfully loaded sybil_attack_log.json with keys: {list(sybil_log_data.keys())}")
        else:
            log_message(f"Sybil attack log file not found at {sybil_log_path}. Proceeding without Sybil log data.")
            # It's critical for this model, so we might want to return False if not found
            # For now, let's allow it to proceed and let data preparation handle missing sybil data
            # return False 
    except Exception as e:
        log_message(f"Error reading sybil_attack_log.json: {e}")
        # return False # Critical error

    # Extract node addresses
    node_addresses = []
    
    # Add addresses from demo_context.json (legitimate nodes)
    if context: # Check if context was loaded
        if "deployerAddress" in context:
            node_addresses.append(context["deployerAddress"])
        if "manufacturerAddress" in context:
            node_addresses.append(context["manufacturerAddress"])
        for i in range(1, 4):
            key = f"transporter{i}Address"
            if key in context:
                node_addresses.append(context[key])
        for role in ["retailerAddress", "buyer1Address", "buyer2Address", "arbitratorAddress"]:
            if role in context:
                node_addresses.append(context[role])
    
    log_message(f"Found {len(node_addresses)} initial node addresses from demo_context.json")

    # Add Sybil node addresses from sybil_attack_log.json
    sybil_node_addresses = []
    if sybil_log_data and "sybilNodes" in sybil_log_data:
        for sybil_node_info in sybil_log_data["sybilNodes"]:
            if "address" in sybil_node_info:
                sybil_node_addresses.append(sybil_node_info["address"])
        log_message(f"Found {len(sybil_node_addresses)} Sybil node addresses from sybil_attack_log.json: {sybil_node_addresses}")
        # Add to the main list, ensuring no duplicates if some sybils were also in demo_context (unlikely for this setup)
        for addr in sybil_node_addresses:
            if addr not in node_addresses:
                node_addresses.append(addr)
    
    # If no addresses found at all, use placeholders (fallback, ideally should have data)
    if not node_addresses:
        log_message("No node addresses found in context or sybil_attack_log. Using hardcoded placeholders.")
        node_addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
        ]
        log_message(f"Using {len(node_addresses)} placeholder addresses.")

    log_message(f"Total {len(node_addresses)} unique node addresses for Sybil Detection: {node_addresses}")
    
    if not sybil_log_data:
        log_message("CRITICAL: sybil_attack_log.json is missing or failed to load. This model relies on it for labeling Sybil nodes.")
        # Depending on strictness, might return False here.
        # For now, we'll let it proceed, but make_federated_data_sybil_real will need to handle this.

    # Check if we have enough data to proceed
    if len(node_addresses) < 2:
        log_message("Warning: Not enough node addresses for meaningful Sybil Detection. Adding placeholders.")
        additional_addresses = [
            "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
            "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"
        ]
        for addr in additional_addresses:
            if addr not in node_addresses:
                node_addresses.append(addr)
        log_message(f"Added placeholder addresses. Total nodes: {len(node_addresses)}")
    
    # Run the Federated Learning model
    try:
        num_clients = min(len(node_addresses), 3) # Use at most 3 clients for demo
        if num_clients == 0 and len(node_addresses) > 0: # Ensure at least one client if nodes exist
            num_clients = 1
            
        log_message(f"Running Sybil Detection with {num_clients} FL clients using {len(node_addresses)} total nodes.")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_sybil_real(
            all_node_addresses=node_addresses, # Pass all addresses
            num_fl_clients=num_clients,
            sybil_attack_log=sybil_log_data # Pass the loaded sybil log data
        )
        
        if not federated_data or all(len(client_data.as_numpy_iterator().next()[0]) == 0 if len(list(client_data)) > 0 else True for client_data in federated_data): # Check if federated_data is empty or contains no actual data
            log_message("Federated data preparation returned empty or insufficient data. Aborting.")
            return False

        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Sybil Detection",
            "num_clients": num_clients,
            "node_addresses_processed": node_addresses,
            "datasets_prepared": len(federated_data),
            "contract_address": os.getenv("CONTRACT_ADDRESS"),
            "rpc_url": os.getenv("POLYGON_AMOY_RPC"),
            "sybil_log_used": sybil_log_path if sybil_log_data else "Not found or error loading"
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "sybil_detection_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Sybil Detection results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Sybil Detection: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Sybil Detection FL Model ===")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
