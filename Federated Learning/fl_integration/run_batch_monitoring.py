#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

# Add FL Model directory to path - adjusted for new directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_Model"))
from tff_batch_monitoring.real_data_preparation_batch_monitoring import make_federated_data_batch_monitoring_real

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def main():
    log_message("=== Starting Batch Monitoring FL Model ===")
    
    # Read demo_context.json to get batch data - adjusted for new directory structure
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

    # Load sybil_attack_log.json
    sybil_log_data = {}
    try:
        sybil_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json")
        log_message(f"Looking for Sybil attack log file at: {sybil_log_path}")
        if os.path.exists(sybil_log_path):
            with open(sybil_log_path, "r", encoding='utf-8') as f:
                sybil_log_data = json.load(f)
            log_message(f"Successfully loaded sybil_attack_log.json with keys: {list(sybil_log_data.keys())}")
        else:
            log_message(f"Sybil attack log file not found at {sybil_log_path}. Proceeding without it.")
    except Exception as e:
        log_message(f"Error reading sybil_attack_log.json: {e}. Proceeding without it.")
        sybil_log_data = {}
    
    # Extract batch-related data from context
    batch_data = []
    
    # Extract product details which would be used in batches
    if "productDetails" in context:
        log_message(f"Found {len(context['productDetails'])} products in context")
        for product in context["productDetails"]:
            if "tokenId" in product and "batchNumber" in product:
                batch_data.append({
                    "tokenId": product["tokenId"],
                    "batchNumber": product["batchNumber"],
                    "productType": product.get("productType", "Unknown"),
                    "manufacturerID": product.get("manufacturerID", "Unknown")
                })
                log_message(f"Added product to batch data: Token ID {product['tokenId']}, Batch {product['batchNumber']}")
    
    # If no batch data found in context, create some placeholder data
    if len(batch_data) == 0:
        log_message("No batch data found in context, creating placeholder data")
        batch_data = [
            {"tokenId": "1", "batchNumber": "B_ALPHA_001", "productType": "Electronics", "manufacturerID": "MANU_ACME_CORP"},
            {"tokenId": "2", "batchNumber": "B_BETA_002", "productType": "Pharmaceuticals", "manufacturerID": "MANU_HEALTHCARE_INC"},
            {"tokenId": "3", "batchNumber": "B_GAMMA_003", "productType": "Luxury Goods", "manufacturerID": "MANU_FASHION_LUXE"}
        ]
        log_message(f"Created {len(batch_data)} placeholder batch entries")
    
    # Extract node addresses for FL clients
    node_addresses = []
    
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

    # Add retailer addresses if available
    for i in range(1, 4): # Assuming up to 3 retailers
        key = f"retailer{i}Address"
        if key in context:
            node_addresses.append(context[key])
            log_message(f"Added retailer address: {context[key]}")
            
    # Add Sybil node addresses from sybil_attack_log.json
    if "sybilNodes" in sybil_log_data:
        for sybil_node in sybil_log_data["sybilNodes"]:
            if "address" in sybil_node and sybil_node["address"] not in node_addresses:
                node_addresses.append(sybil_node["address"])
                log_message(f"Added Sybil node address from log: {sybil_node['address']}")
    
    # If no addresses found, use placeholder addresses
    if len(node_addresses) == 0:
        log_message("No node addresses found in context, using placeholder addresses")
        node_addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # Example address 1
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Example address 2
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906"   # Example address 3
        ]
        log_message(f"Using {len(node_addresses)} placeholder addresses")
    
    # Run the Federated Learning model
    try:
        num_clients = min(len(node_addresses), 3)  # Use at most 3 clients for demo
        log_message(f"Running Batch Monitoring with {num_clients} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_batch_monitoring_real(node_addresses, num_clients, sybil_log_data)
        
        # Here you would normally run the actual FL training
        # For demo purposes, we'll just log the data preparation success
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Batch Monitoring",
            "num_clients": num_clients,
            "node_addresses": node_addresses,
            "batch_data": batch_data,
            "datasets_prepared": len(federated_data)
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "batch_monitoring_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Batch Monitoring results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Batch Monitoring: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Batch Monitoring FL Model ===")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
