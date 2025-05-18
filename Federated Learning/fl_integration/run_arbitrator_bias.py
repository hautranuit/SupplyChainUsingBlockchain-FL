#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

# Add FL Model directory to path - adjusted for new directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_Model"))
from tff_advanced_analysis.arbitrator_bias.real_data_preparation_arbitrator_bias import make_federated_data_arbitrator_bias_real

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def main():
    log_message("=== Starting Arbitrator Bias FL Model ===")
    
    # Read demo_context.json to get dispute and arbitrator data - adjusted for new directory structure
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
    
    # Extract arbitrator and dispute data
    arbitrator_data = []
    
    # Check if arbitrator address is in context
    arbitrator_address = context.get("arbitratorAddress", None)
    if arbitrator_address:
        log_message(f"Found arbitrator address in context: {arbitrator_address}")
        
        # In a real implementation, you would extract dispute resolution history
        # For demo purposes, we'll create a placeholder dispute resolution record
        arbitrator_data.append({
            "arbitrator_address": arbitrator_address,
            "dispute_id": "1",
            "resolution_outcome": "FavorPlaintiff",  # Example: FavorPlaintiff, FavorDefendant, Partial, Dismissed
            "plaintiff_address": context.get("buyer1Address", "0x1234..."),
            "defendant_address": context.get("manufacturerAddress", "0x5678..."),
            "dispute_type": "ProductQuality",
            "resolution_timestamp": datetime.now().isoformat(),
            "resolution_details": "Arbitrator decision: Partial refund issued, product to be returned."
        })
        log_message(f"Added dispute resolution record for arbitrator {arbitrator_address}")
    else:
        log_message("No arbitrator address found in context")
    
    # If no arbitrator data found, create placeholder data
    if len(arbitrator_data) == 0:
        log_message("No arbitrator data found, creating placeholder data")
        arbitrator_data = [
            {
                "arbitrator_address": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
                "dispute_id": "1",
                "resolution_outcome": "FavorPlaintiff",
                "plaintiff_address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                "defendant_address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "dispute_type": "ProductQuality",
                "resolution_timestamp": datetime.now().isoformat(),
                "resolution_details": "Arbitrator decision: Full refund issued."
            },
            {
                "arbitrator_address": "0x8626f6940E2eb28930eFb4CeF49B2d1F2C9C1199",
                "dispute_id": "2",
                "resolution_outcome": "FavorDefendant",
                "plaintiff_address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                "defendant_address": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
                "dispute_type": "DeliveryDelay",
                "resolution_timestamp": datetime.now().isoformat(),
                "resolution_details": "Arbitrator decision: Claim dismissed, delay was justified."
            }
        ]
        log_message(f"Created {len(arbitrator_data)} placeholder arbitrator records")
    
    # Extract node addresses for FL clients
    node_addresses = []
    
    # Add arbitrator address
    if arbitrator_address:
        node_addresses.append(arbitrator_address)
    
    # Add other addresses from context for FL clients
    for role in ["manufacturerAddress", "transporter1Address", "retailerAddress", "buyer1Address"]:
        if role in context:
            node_addresses.append(context[role])
    
    # If not enough addresses, add placeholder addresses
    if len(node_addresses) < 3:
        placeholder_addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
        ]
        for addr in placeholder_addresses:
            if addr not in node_addresses:
                node_addresses.append(addr)
                if len(node_addresses) >= 3:
                    break
    
    log_message(f"Using {len(node_addresses)} node addresses for FL clients")
    
    # Run the Federated Learning model
    try:
        num_clients = min(len(node_addresses), 3)  # Use at most 3 clients for demo
        log_message(f"Running Arbitrator Bias analysis with {num_clients} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_arbitrator_bias_real(arbitrator_data, node_addresses, num_fl_clients=num_clients)
        
        # Here you would normally run the actual FL training
        # For demo purposes, we'll just log the data preparation success
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Arbitrator Bias",
            "num_clients": num_clients,
            "node_addresses": node_addresses,
            "arbitrator_data_summary": [{"arbitrator": record["arbitrator_address"], "dispute_id": record["dispute_id"], "outcome": record["resolution_outcome"]} for record in arbitrator_data],
            "datasets_prepared": len(federated_data)
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "arbitrator_bias_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Arbitrator Bias results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Arbitrator Bias analysis: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Arbitrator Bias FL Model ===")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
