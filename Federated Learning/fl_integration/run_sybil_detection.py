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
    
    # Read demo_context.json to get node addresses - adjusted for new directory structure
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
                log_message("Context file not found at alternative path either")
                return False
        
        with open(context_path, "r", encoding='utf-8') as f:
            context = json.load(f)
            log_message(f"Successfully loaded context with keys: {list(context.keys())}")
    except Exception as e:
        log_message(f"Error reading demo context: {e}")
        return False
    
    # Extract node addresses from context
    node_addresses = []
    
    # Add deployer/admin address
    if "deployerAddress" in context:
        node_addresses.append(context["deployerAddress"])
        log_message(f"Added deployer address: {context['deployerAddress']}")
    
    # Add manufacturer address
    if "manufacturerAddress" in context:
        node_addresses.append(context["manufacturerAddress"])
        log_message(f"Added manufacturer address: {context['manufacturerAddress']}")
    
    # Add transporter addresses if available
    for i in range(1, 4):  # Assuming up to 3 transporters
        key = f"transporter{i}Address"
        if key in context:
            node_addresses.append(context[key])
            log_message(f"Added transporter address: {context[key]}")
    
    # Add retailer, buyer addresses if available
    for role in ["retailerAddress", "buyer1Address", "buyer2Address", "arbitratorAddress"]:
        if role in context:
            node_addresses.append(context[role])
            log_message(f"Added {role}: {context[role]}")
    
    # If no addresses found in context, use addresses from contract deployment
    if len(node_addresses) == 0 and "contractAddress" in context:
        log_message("No node addresses found in context, using signers from hardhat config")
        # In a real implementation, you would query the blockchain for verified nodes
        # For demo purposes, we'll use some placeholder addresses
        node_addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # Example address 1
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Example address 2
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906"   # Example address 3
        ]
        log_message(f"Using {len(node_addresses)} placeholder addresses")
    
    log_message(f"Found {len(node_addresses)} node addresses for Sybil Detection")
    
    # Check if we have enough data to proceed
    if len(node_addresses) < 2:
        log_message("Warning: Not enough node addresses for meaningful Sybil Detection")
        # For demo purposes, add some placeholder addresses to ensure we have enough
        additional_addresses = [
            "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
            "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc"
        ]
        node_addresses.extend(additional_addresses)
        log_message(f"Added {len(additional_addresses)} placeholder addresses for demo purposes")
    
    # Run the Federated Learning model
    try:
        num_clients = min(len(node_addresses), 3)  # Use at most 3 clients for demo
        log_message(f"Running Sybil Detection with {num_clients} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_sybil_real(
            node_addresses, 
            num_fl_clients=num_clients
        )
        
        # Here you would normally run the actual FL training
        # For demo purposes, we'll just log the data preparation success
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Sybil Detection",
            "num_clients": num_clients,
            "node_addresses": node_addresses,
            "datasets_prepared": len(federated_data),
            "contract_address": os.getenv("CONTRACT_ADDRESS"),
            "rpc_url": os.getenv("POLYGON_AMOY_RPC")
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
