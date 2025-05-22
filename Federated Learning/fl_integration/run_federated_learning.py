#!/usr/bin/env python3
"""
Consolidated Federated Learning Training Script for ChainFLIP Supply Chain System

This script replaces multiple individual FL model scripts with a single entry point
that can train any of the supported models based on parameters.

Usage:
    python run_federated_learning.py --model [model_name] [--options]

Models:
    - sybil_detection: Detects Sybil nodes in the network
    - batch_monitoring: Monitors batch processing for anomalies
    - node_behavior: Analyzes node behavior patterns over time
    - dispute_risk: Predicts dispute risks in transactions
    - all: Run all models in sequence (default)

Options:
    --num_clients: Number of FL clients to use (default: auto-determined)
    --debug: Enable debug mode with additional logging
    --attack_mode: Whether to run in attack detection mode (default: auto-detect)
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add FL Model directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_Model"))

# Import data preparation functions for each model
from tff_sybil_detection.real_data_preparation_sybil import make_federated_data_sybil_real
from tff_batch_monitoring.real_data_preparation_batch_monitoring import make_federated_data_batch_monitoring_real
from tff_advanced_analysis.node_behavior_timeseries.real_data_preparation_node_behavior import make_federated_data_p3_timeseries_real
from tff_advanced_analysis.dispute_risk.real_data_preparation_dispute_risk import make_federated_data_p3_dispute_real # Corrected import alias

# Constants
SUPPORTED_MODELS = ["sybil_detection", "batch_monitoring", "node_behavior", "dispute_risk", "all"]
DEFAULT_MODEL = "all"
DEFAULT_NUM_CLIENTS = 3  # Default number of FL clients if not auto-determined
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def log_message(message: str, debug: bool = False) -> None:
    """
    Log a message to both console and log file
    
    Args:
        message: Message to log
        debug: Whether this is a debug message (only logged if debug mode is enabled)
    """
    global args
    if debug and not args.debug:
        return
    
    timestamp = datetime.now().isoformat()
    formatted_message = f"[{timestamp}] {'[DEBUG] ' if debug else ''}{message}"
    print(formatted_message)
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    with open(LOG_FILE, "a", encoding='utf-8') as log_file:
        log_file.write(f"{formatted_message}\n")

def find_file(filename: str, search_paths: List[str]) -> Optional[str]:
    """
    Find a file in various possible locations
    
    Args:
        filename: Name of the file to find
        search_paths: List of paths to search
        
    Returns:
        Path to file if found, None otherwise
    """
    for path in search_paths:
        log_message(f"Looking for {filename} at: {path}", debug=True)
        if os.path.exists(path):
            return path
    
    return None

def find_context_file() -> Optional[str]:
    """
    Find the demo_context.json file in various possible locations
    
    Returns:
        Path to context file if found, None otherwise
    """
    possible_paths = [
        # Standard path (relative to project root)
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                    "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"),
        # Alternative path (in Federated Learning directory)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "demo_context.json"),
        # Another alternative path
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "..", "SupplyChain_dapp", "scripts", "lifecycle_demo", "demo_context.json")
    ]
    
    return find_file("demo_context.json", possible_paths)

def find_sybil_log_file() -> Optional[str]:
    """
    Find the sybil_attack_log.json file in various possible locations
    
    Returns:
        Path to sybil log file if found, None otherwise
    """
    possible_paths = [
        # Standard path (relative to project root)
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                    "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json"),
        # Alternative path (in Federated Learning directory)
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "sybil_attack_log.json"),
        # Another alternative path
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "..", "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json")
    ]
    
    return find_file("sybil_attack_log.json", possible_paths)

def load_json_file(file_path: str, description: str) -> Dict[str, Any]:
    """
    Load data from a JSON file
    
    Args:
        file_path: Path to the JSON file
        description: Description of the file for logging
        
    Returns:
        Dictionary containing file data, empty dict if file not found or invalid
    """
    if not file_path:
        log_message(f"{description} file not found at any expected location")
        return {}
    
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        log_message(f"Successfully loaded {description} with keys: {list(data.keys())}")
        return data
    except Exception as e:
        log_message(f"Error reading {description}: {e}")
        return {}

def load_context_data() -> Dict[str, Any]:
    """
    Load data from demo_context.json
    
    Returns:
        Dictionary containing context data, empty dict if file not found or invalid
    """
    context_path = find_context_file()
    return load_json_file(context_path, "Context file (demo_context.json)")

def load_sybil_log_data() -> Dict[str, Any]:
    """
    Load data from sybil_attack_log.json
    
    Returns:
        Dictionary containing sybil log data, empty dict if file not found or invalid
    """
    sybil_log_path = find_sybil_log_file()
    return load_json_file(sybil_log_path, "Sybil attack log (sybil_attack_log.json)")

def detect_attack_mode(sybil_log_data: Dict[str, Any]) -> bool:
    """
    Detect if we're in attack mode based on sybil log data
    
    Args:
        sybil_log_data: Sybil log data from sybil_attack_log.json
        
    Returns:
        True if attack mode detected, False otherwise
    """
    if not sybil_log_data:
        return False
        
    # Check if there are any sybil nodes defined
    if "sybilNodes" in sybil_log_data and sybil_log_data["sybilNodes"]:
        return True
        
    # Check if any scenario has actions
    for scenario in ["scenarioA", "scenarioB", "scenarioC", "scenarioD"]:
        if scenario in sybil_log_data and "actions" in sybil_log_data[scenario] and sybil_log_data[scenario]["actions"]:
            return True
            
    # Check for bribed nodes in scenario D
    if "scenarioD" in sybil_log_data and "bribedNodes" in sybil_log_data["scenarioD"] and sybil_log_data["scenarioD"]["bribedNodes"]:
        return True
            
    return False

def extract_node_addresses(context: Dict[str, Any], sybil_log_data: Dict[str, Any]) -> List[str]:
    """
    Extract node addresses from context and sybil log data
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        
    Returns:
        List of node addresses
    """
    node_addresses = []
    
    # Extract from context
    address_keys = [
        "deployerAddress", "manufacturerAddress", 
        "transporter1Address", "transporter2Address", "transporter3Address",
        "retailerAddress", "retailer1Address", "retailer2Address", "retailer3Address",
        "buyer1Address", "buyer2Address", "buyer3Address", 
        "arbitratorAddress"
    ]
    
    for key in address_keys:
        if key in context and context[key] not in node_addresses:
            node_addresses.append(context[key])
            log_message(f"Added node address from context: {context[key]} ({key})", debug=True)
    
    # Extract Sybil node addresses
    if sybil_log_data and "sybilNodes" in sybil_log_data:
        for sybil_node in sybil_log_data["sybilNodes"]:
            if "address" in sybil_node and sybil_node["address"] not in node_addresses:
                node_addresses.append(sybil_node["address"])
                log_message(f"Added Sybil node address: {sybil_node['address']}", debug=True)
    
    # Extract Bribed node addresses (Scenario D)
    if sybil_log_data and "scenarioD" in sybil_log_data and "bribedNodes" in sybil_log_data["scenarioD"]:
        for bribed_node in sybil_log_data["scenarioD"]["bribedNodes"]:
            if "address" in bribed_node and bribed_node["address"] not in node_addresses:
                node_addresses.append(bribed_node["address"])
                log_message(f"Added Bribed node address: {bribed_node['address']}", debug=True)
    
    # If no addresses found, use placeholder addresses
    if not node_addresses:
        log_message("No node addresses found in context or sybil log. Using placeholder addresses.")
        node_addresses = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
        ]
    
    log_message(f"Total unique node addresses: {len(node_addresses)}")
    return node_addresses

def extract_batch_data(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract batch data from context
    
    Args:
        context: Context data from demo_context.json
        
    Returns:
        List of batch data dictionaries
    """
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
                log_message(f"Added product to batch data: Token ID {product['tokenId']}, Batch {product['batchNumber']}", debug=True)
    
    # If no batch data found, create placeholder data
    if not batch_data:
        log_message("No batch data found in context, creating placeholder data")
        batch_data = [
            {"tokenId": "1", "batchNumber": "B_ALPHA_001", "productType": "Electronics", "manufacturerID": "MANU_ACME_CORP"},
            {"tokenId": "2", "batchNumber": "B_BETA_002", "productType": "Pharmaceuticals", "manufacturerID": "MANU_HEALTHCARE_INC"},
            {"tokenId": "3", "batchNumber": "B_GAMMA_003", "productType": "Luxury Goods", "manufacturerID": "MANU_FASHION_LUXE"}
        ]
    
    return batch_data

def extract_transaction_data(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract transaction data from context for dispute risk analysis
    
    Args:
        context: Context data from demo_context.json
        
    Returns:
        List of transaction data dictionaries
    """
    transaction_data = []
    
    # Check if product details are in context
    if "productDetails" in context:
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
                    "price": product.get("price", "0"),
                    "timestamp": datetime.now().isoformat(),
                    "has_dispute": False,
                    "dispute_reason": None
                }
                
                # Check if this product has a dispute
                if product.get("productStatus") and "dispute" in product.get("productStatus", "").lower():
                    transaction_record["has_dispute"] = True
                    transaction_record["dispute_reason"] = "Product received damaged, quality not as expected."
                
                transaction_data.append(transaction_record)
    
    # If no transaction data found, create placeholder data
    if not transaction_data:
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
    
    return transaction_data

def prepare_node_data_for_timeseries(context: Dict[str, Any], sybil_log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare node data for timeseries analysis
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        
    Returns:
        List of node data dictionaries with address, role, and activities
    """
    node_data_for_fl = []
    all_node_addresses = set()

    # Helper to add node if address is present
    def add_node_if_present(address, role_name):
        if address and address not in all_node_addresses:
            node_data_for_fl.append({
                "address": address,
                "role": role_name,
                "activities": []  # Activities will be populated by data prep script
            })
            all_node_addresses.add(address)
            log_message(f"Added {role_name} ({address}) to FL client list", debug=True)

    # Extract from demo_context.json
    roles_to_extract = [
        ("manufacturerAddress", "Manufacturer"),
        ("transporter1Address", "Transporter1"),
        ("transporter2Address", "Transporter2"),
        ("transporter3Address", "Transporter3"),
        ("retailerAddress", "Retailer"),
        ("buyer1Address", "Buyer1"),
        ("buyer2Address", "Buyer2"),
        ("arbitratorAddress", "Arbitrator")
    ]
    for context_key, role_name in roles_to_extract:
        if context_key in context:
            add_node_if_present(context[context_key], role_name)

    # Extract Sybil node addresses from sybil_attack_log.json
    if sybil_log_data and "sybilNodes" in sybil_log_data:
        for sybil_node_info in sybil_log_data["sybilNodes"]:
            sybil_address = sybil_node_info.get("address")
            add_node_if_present(sybil_address, f"SybilNode_{sybil_node_info.get('id', '')}")
    
    # Extract Bribed node addresses from sybil_attack_log.json (Scenario D)
    if sybil_log_data and "scenarioD" in sybil_log_data and "bribedNodes" in sybil_log_data["scenarioD"]:
        for bribed_node_info in sybil_log_data["scenarioD"]["bribedNodes"]:
            bribed_address = bribed_node_info.get("address")
            bribed_role = bribed_node_info.get("role", "BribedNode")
            add_node_if_present(bribed_address, f"Bribed_{bribed_role}")

    return node_data_for_fl

def run_sybil_detection(context: Dict[str, Any], sybil_log_data: Dict[str, Any], num_clients: int) -> bool:
    """
    Run Sybil Detection FL model
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        num_clients: Number of FL clients to use
        
    Returns:
        True if successful, False otherwise
    """
    log_message("=== Starting Sybil Detection FL Model ===")
    
    try:
        # Extract node addresses
        node_addresses = extract_node_addresses(context, sybil_log_data)
        
        if not node_addresses:
            log_message("No node addresses found for Sybil Detection. Cannot proceed.")
            return False
        
        # Determine number of clients to use
        num_clients_to_use = min(len(node_addresses), num_clients) if num_clients > 0 else min(len(node_addresses), DEFAULT_NUM_CLIENTS)
        if num_clients_to_use == 0 and len(node_addresses) > 0:
            num_clients_to_use = 1
            
        log_message(f"Running Sybil Detection with {num_clients_to_use} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_sybil_real(
            all_node_addresses=node_addresses,
            num_fl_clients=num_clients_to_use,
            sybil_attack_log=sybil_log_data
        )
        
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Sybil Detection",
            "num_clients": num_clients_to_use,
            "node_addresses": node_addresses,
            "datasets_prepared": len(federated_data),
            "attack_mode": args.attack_mode or detect_attack_mode(sybil_log_data)
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "sybil_detection_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Sybil Detection results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Sybil Detection: {e}")
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Sybil Detection FL Model ===")

def run_batch_monitoring(context: Dict[str, Any], sybil_log_data: Dict[str, Any], num_clients: int) -> bool:
    """
    Run Batch Monitoring FL model
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        num_clients: Number of FL clients to use
        
    Returns:
        True if successful, False otherwise
    """
    log_message("=== Starting Batch Monitoring FL Model ===")
    
    try:
        # Extract node addresses and batch data
        node_addresses = extract_node_addresses(context, sybil_log_data)
        batch_data = extract_batch_data(context)
        
        if not node_addresses:
            log_message("No node addresses found for Batch Monitoring. Cannot proceed.")
            return False
        
        # Determine number of clients to use
        num_clients_to_use = min(len(node_addresses), num_clients) if num_clients > 0 else min(len(node_addresses), DEFAULT_NUM_CLIENTS)
        if num_clients_to_use == 0 and len(node_addresses) > 0:
            num_clients_to_use = 1
            
        log_message(f"Running Batch Monitoring with {num_clients_to_use} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_batch_monitoring_real(
            all_node_addresses=node_addresses,
            num_fl_clients=num_clients_to_use,
            sybil_attack_log=sybil_log_data
        )
        
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Batch Monitoring",
            "num_clients": num_clients_to_use,
            "node_addresses": node_addresses,
            "batch_data_summary": [{"tokenId": batch["tokenId"], "batchNumber": batch["batchNumber"]} for batch in batch_data],
            "datasets_prepared": len(federated_data),
            "attack_mode": args.attack_mode or detect_attack_mode(sybil_log_data)
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "batch_monitoring_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Batch Monitoring results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Batch Monitoring: {e}")
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Batch Monitoring FL Model ===")

def run_node_behavior_timeseries(context: Dict[str, Any], sybil_log_data: Dict[str, Any], num_clients: int) -> bool:
    """
    Run Node Behavior Timeseries FL model
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        num_clients: Number of FL clients to use
        
    Returns:
        True if successful, False otherwise
    """
    log_message("=== Starting Node Behavior Timeseries FL Model ===")
    
    try:
        # Prepare node data for timeseries analysis
        node_data_for_fl = prepare_node_data_for_timeseries(context, sybil_log_data)
        
        if not node_data_for_fl:
            log_message("No node data found for Node Behavior Timeseries. Cannot proceed.")
            return False
        
        # Determine number of clients to use
        num_clients_to_use = min(len(node_data_for_fl), num_clients) if num_clients > 0 else len(node_data_for_fl)
        log_message(f"Running Node Behavior Timeseries with {num_clients_to_use} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_p3_timeseries_real(
            clients_info=node_data_for_fl,
            sybil_attack_log=sybil_log_data
        )
        
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients for Node Behavior Timeseries")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Node Behavior Timeseries",
            "num_clients": num_clients_to_use,
            "node_addresses": [node["address"] for node in node_data_for_fl],
            "datasets_prepared": len(federated_data),
            "attack_mode": args.attack_mode or detect_attack_mode(sybil_log_data)
        }
        
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "node_behavior_timeseries_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Node Behavior Timeseries results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Node Behavior Timeseries: {e}")
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Node Behavior Timeseries FL Model ===")

def run_dispute_risk(context: Dict[str, Any], sybil_log_data: Dict[str, Any], num_clients: int) -> bool:
    """
    Run Dispute Risk FL model
    
    Args:
        context: Context data from demo_context.json
        sybil_log_data: Sybil log data from sybil_attack_log.json
        num_clients: Number of FL clients to use
        
    Returns:
        True if successful, False otherwise
    """
    log_message("=== Starting Dispute Risk FL Model ===")
    
    try:
        # Extract node addresses and transaction data
        node_addresses = extract_node_addresses(context, sybil_log_data)
        transaction_data = extract_transaction_data(context)
        
        if not node_addresses:
            log_message("No node addresses found for Dispute Risk. Cannot proceed.")
            return False
        
        # Determine number of clients to use
        num_clients_to_use = min(len(node_addresses), num_clients) if num_clients > 0 else min(len(node_addresses), DEFAULT_NUM_CLIENTS)
        if num_clients_to_use == 0 and len(node_addresses) > 0:
            num_clients_to_use = 1
            
        log_message(f"Running Dispute Risk with {num_clients_to_use} FL clients")
        
        # Make federated datasets
        log_message("Preparing federated data...")
        federated_data = make_federated_data_p3_dispute_real(
            all_node_addresses=node_addresses,
            sybil_attack_log=sybil_log_data
        )
        
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients for Dispute Risk")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Dispute Risk",
            "num_clients": num_clients_to_use,
            "node_addresses": node_addresses,
            "datasets_prepared": len(federated_data),
            "attack_mode": args.attack_mode or detect_attack_mode(sybil_log_data)
        }

        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "dispute_risk_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        log_message(f"Dispute Risk results saved to {results_path}")
        return True
        
    except Exception as e:
        log_message(f"Error running Dispute Risk: {e}")
        log_message(traceback.format_exc())
        return False
    finally:
        log_message("=== Completed Dispute Risk FL Model ===")

def main() -> None:
    """
    Main entry point for the script
    """
    global args
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run Federated Learning models for ChainFLIP")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=SUPPORTED_MODELS,
                        help="FL model to run (default: all)")
    parser.add_argument("--num_clients", type=int, default=0,
                        help="Number of FL clients to use (default: auto-determined)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with additional logging")
    parser.add_argument("--attack_mode", type=str, default="auto",
                        help="Attack mode detection (default: auto-detect)")
    args = parser.parse_args()
    
    # Log startup
    log_message("=== Starting Federated Learning Run ===")
    log_message(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Load context and sybil log data
    context_data = load_context_data()
    sybil_log_data = load_sybil_log_data()
    
    # Detect attack mode if not explicitly set
    if args.attack_mode == "auto":
        args.attack_mode = detect_attack_mode(sybil_log_data)
        log_message(f"Auto-detected attack mode: {args.attack_mode}")
    
    # Model selection and execution
    model_functions = {
        "sybil_detection": run_sybil_detection,
        "batch_monitoring": run_batch_monitoring,
        "node_behavior": run_node_behavior_timeseries,
        "dispute_risk": run_dispute_risk
    }
    
    if args.model in model_functions:
        model_func = model_functions[args.model]
        success = model_func(context_data, sybil_log_data, args.num_clients)
        
        if success:
            log_message(f"Model {args.model} completed successfully")
        else:
            log_message(f"Model {args.model} failed")
    elif args.model == "all":
        for model_name, model_func in model_functions.items():
            log_message(f"=== Running model: {model_name} ===")
            success = model_func(context_data, sybil_log_data, args.num_clients)
            if success:
                log_message(f"Model {model_name} completed successfully")
            else:
                log_message(f"Model {model_name} failed")
    else:
        log_message(f"Unsupported model specified: {args.model}")
    
    # Log completion
    log_message("=== Completed Federated Learning Run ===")

if __name__ == "__main__":
    main()
