#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

# Add FL Model directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Federated Learning/FL_Model"))
from tff_advanced_analysis.dispute_risk.real_data_preparation_dispute_risk import make_federated_data_dispute_risk_real

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def main():
    log_message("=== Starting Dispute Risk FL Model ===")
    
    # Read demo_context.json to get transaction and product data
    try:
        context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json")
        log_message(f"Looking for context file at: {context_path}")
        
        if not os.path.exists(context_path):
            log_message(f"Context file not found at {context_path}")
            return False
        
        with open(context_path, "r") as f:
            context = json.load(f)
            log_message(f"Successfully loaded context with keys: {list(context.keys())}")
    except Exception as e:
        log_message(f"Error reading demo context: {e}")
        return False
    
    # Extract transaction and product data for dispute risk analysis
    transaction_data = []
    
    # Check if product details are in context
    if "productDetails" in context:
        log_message(f"Found {len(context['productDetails'])} products in context")
        
        for product in context["productDetails"]:
            # Extract transaction data from product details
            if "tokenId" in product and "currentOwnerAddress" in product:
                # Create transaction record
                transaction_record = {
                    "tokenId": product["tokenId"],
                    "productType": product.get("productType", "Unknown"),
                    "manufacturerID": product.get("manufacturerID", "Unknown"),
                    "buyer": product["currentOwnerAddress"],
                    "seller": context.get("manufacturerAddress", "Unknown"),
                    "price": product.get("purchasePrice", "0"),
                    "timestamp": datetime.now().isoformat(),
                    "has_dispute": False,
                    "dispute_reason": None
                }
                
                # Check if this product has a dispute
                # In a real implementation, you would check for actual dispute records
                # For demo purposes, we'll assume the first product has a dispute
                if product["tokenId"] == context["productDetails"][0]["tokenId"]:
                    transaction_record["has_dispute"] = True
                    transaction_record["dispute_reason"] = "Product received damaged, quality not as expected."
                    log_message(f"Added dispute flag to product {product['tokenId']}")
                
                transaction_data.append(transaction_record)
                log_message(f"Added transaction record for token ID {product['tokenId']}")
    
    # If no transaction data found, create placeholder data
    if len(transaction_data) == 0:
        log_message("No transaction data found, creating placeholder data")
        transaction_data = [
            {
                "tokenId": "1",
                "productType": "Electronics",
                "manufacturerID": "MANU_ACME_CORP",
                "buyer": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                "seller": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "price": "100000000000000000",  # 0.1 ETH in wei
                "timestamp": datetime.now().isoformat(),
                "has_dispute": True,
                "dispute_reason": "Product received damaged, quality not as expected."
            },
            {
                "tokenId": "2",
                "productType": "Pharmaceuticals",
                "manufacturerID": "MANU_HEALTHCARE_INC",
                "buyer": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
                "seller": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "price": "50000000000000000",  # 0.05 ETH in wei
                "timestamp": datetime.now().isoformat(),
                "has_dispute": False,
                "dispute_reason": None
            }
        ]
        log_message(f"Created {len(transaction_data)} placeholder transaction records")
    
    # Extract node addresses for FL clients
    node_addresses = []
    
    # Add addresses from context for FL clients
    for role in ["manufacturerAddress", "transporter1Address", "retailerAddress", "buyer1Address", "arbitratorAddress"]:
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
        log_message(f"Running Dispute Risk analysis with {num_clients} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_dispute_risk_real(transaction_data, node_addresses, num_fl_clients=num_clients)
        
        # Here you would normally run the actual FL training
        # For demo purposes, we'll just log the data preparation success
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results to a file for later analysis
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Dispute Risk",
            "num_clients": num_clients,
            "node_addresses": node_addresses,
            "transaction_data_summary": [{"tokenId": tx["tokenId"], "productType": tx["productType"], "has_dispute": tx["has_dispute"]} for tx in transaction_data],
            "datasets_prepared": len(federated_data)
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, "dispute_risk_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Dispute Risk results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Dispute Risk analysis: {e}")
        import traceback
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Dispute Risk FL Model ===")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
