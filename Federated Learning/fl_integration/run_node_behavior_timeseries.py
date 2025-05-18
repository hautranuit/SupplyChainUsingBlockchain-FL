#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

# Add FL Model directory to path - adjusted for new directory structure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_Model"))
from tff_advanced_analysis.node_behavior_timeseries.real_data_preparation_node_behavior import make_federated_data_p3_timeseries_real

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def main():
    log_message("=== Starting Node Behavior Timeseries FL Model ===")
    
    # Read demo_context.json to get node activity data - adjusted for new directory structure
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
    
    # Extract node addresses and their activities
    node_data = []
    
    # Add manufacturer address and activities
    if "manufacturerAddress" in context:
        manufacturer_address = context["manufacturerAddress"]
        manufacturer_activities = []
        
        # Check product details for manufacturer activities
        if "productDetails" in context:
            for product in context["productDetails"]:
                if "manufacturerID" in product:
                    manufacturer_activities.append({
                        "activity_type": "product_minting",
                        "tokenId": product.get("tokenId", "unknown"),
                        "timestamp": datetime.now().isoformat(),
                        "details": f"Minted product {product.get('uniqueProductID', 'unknown')}"
                    })
        
        node_data.append({
            "address": manufacturer_address,
            "role": "Manufacturer",
            "activities": manufacturer_activities
        })
        log_message(f"Added manufacturer with {len(manufacturer_activities)} activities")
    
    # Add transporter addresses and activities
    for i in range(1, 4):  # Assuming up to 3 transporters
        key = f"transporter{i}Address"
        if key in context:
            transporter_address = context[key]
            transporter_activities = []
            
            # Check product details for transport activities
            if "productDetails" in context:
                for product in context["productDetails"]:
                    if "transportStatus" in product and product["transportStatus"] == "Completed":
                        transporter_activities.append({
                            "activity_type": "transport",
                            "tokenId": product.get("tokenId", "unknown"),
                            "timestamp": datetime.now().isoformat(),
                            "details": f"Transported product {product.get('uniqueProductID', 'unknown')}"
                        })
            
            node_data.append({
                "address": transporter_address,
                "role": f"Transporter{i}",
                "activities": transporter_activities
            })
            log_message(f"Added transporter{i} with {len(transporter_activities)} activities")
    
    # Add buyer/retailer addresses and activities
    for role in ["retailerAddress", "buyer1Address", "buyer2Address"]:
        if role in context:
            buyer_address = context[role]
            buyer_activities = []
            
            # Check product details for purchase activities
            if "productDetails" in context:
                for product in context["productDetails"]:
                    if "currentOwnerAddress" in product and product["currentOwnerAddress"] == buyer_address:
                        buyer_activities.append({
                            "activity_type": "purchase",
                            "tokenId": product.get("tokenId", "unknown"),
                            "timestamp": datetime.now().isoformat(),
                            "details": f"Purchased product {product.get('uniqueProductID', 'unknown')}"
                        })
            
            node_data.append({
                "address": buyer_address,
                "role": role.replace("Address", ""),
                "activities": buyer_activities
            })
            log_message(f"Added {role} with {len(buyer_activities)} activities")
    
    # Add arbitrator address and activities
    if "arbitratorAddress" in context:
        arbitrator_address = context["arbitratorAddress"]
        arbitrator_activities = []
        
        # In a real implementation, you would extract dispute resolution activities
        # For demo purposes, we'll add a placeholder activity
        arbitrator_activities.append({
            "activity_type": "arbitration",
            "disputeId": "1",
            "timestamp": datetime.now().isoformat(),
            "details": "Resolved dispute for product"
        })
        
        node_data.append({
            "address": arbitrator_address,
            "role": "Arbitrator",
            "activities": arbitrator_activities
        })
        log_message(f"Added arbitrator with {len(arbitrator_activities)} activities")
    
    # If no node data found, create placeholder data
    if len(node_data) == 0:
        log_message("No node data found in context, creating placeholder data")
        node_data = [
            {
                "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                "role": "Manufacturer",
                "activities": [
                    {"activity_type": "product_minting", "tokenId": "1", "timestamp": datetime.now().isoformat()}
                ]
            },
            {
                "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "role": "Transporter",
                "activities": [
                    {"activity_type": "transport", "tokenId": "1", "timestamp": datetime.now().isoformat()}
                ]
            },
            {
                "address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                "role": "Buyer",
                "activities": [
                    {"activity_type": "purchase", "tokenId": "1", "timestamp": datetime.now().isoformat()}
                ]
            }
        ]
        log_message(f"Created {len(node_data)} placeholder node entries")
    
    # Extract node addresses for FL clients
    node_addresses = [node["address"] for node in node_data]
    log_message(f"Extracted {len(node_addresses)} node addresses for FL clients")
    
    # Run the Federated Learning model
    try:
        num_clients = min(len(node_addresses), 3)  # Use at most 3 clients for demo
        log_message(f"Running Node Behavior Timeseries with {num_clients} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_p3_timeseries_real(node_data, num_fl_clients=num_clients)
        
        # Here you would normally run the actual FL training
        # For demo purposes, we'll just log the data preparation success
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Node Behavior Timeseries",
            "num_clients": num_clients,
            "node_addresses": node_addresses,
            "node_data_summary": [{"address": node["address"], "role": node["role"], "activity_count": len(node["activities"])} for node in node_data],
            "datasets_prepared": len(federated_data)
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "node_behavior_timeseries_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Node Behavior Timeseries results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Node Behavior Timeseries: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Node Behavior Timeseries FL Model ===")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
