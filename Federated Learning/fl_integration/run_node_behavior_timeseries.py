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
    
    # --- Load demo_context.json ---
    try:
        # Primary path for demo_context.json
        context_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"
        )
        log_message(f"Looking for context file at: {context_path}")
        if not os.path.exists(context_path):
            log_message(f"Context file not found at {context_path}. Trying alternative path...")
            # Alternative path (e.g., if fl_integration is at the same level as SupplyChain_dapp)
            context_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"
            )
            log_message(f"Trying alternative path: {context_path}")
            if not os.path.exists(context_path):
                 # Fallback path if script is run from within fl_integration directory directly
                context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "..", "SupplyChain_dapp", "scripts", "lifecycle_demo", "demo_context.json")
                log_message(f"Trying yet another alternative path: {context_path}")
                if not os.path.exists(context_path):
                    log_message("Context file demo_context.json not found at any expected location.")
                    return False
        
        with open(context_path, "r", encoding='utf-8') as f:
            context = json.load(f)
            log_message(f"Successfully loaded demo_context.json with keys: {list(context.keys())}")
    except Exception as e:
        log_message(f"Error reading demo_context.json: {e}")
        return False

    # --- Load sybil_attack_log.json ---
    sybil_log_data = None
    try:
        # Path for sybil_attack_log.json (similar logic to demo_context.json)
        sybil_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json"
        )
        log_message(f"Looking for sybil_attack_log.json at: {sybil_log_path}")
        if not os.path.exists(sybil_log_path):
            log_message(f"Sybil log file not found at {sybil_log_path}. Trying alternative path...")
            sybil_log_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                "SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json"
            )
            log_message(f"Trying alternative path for sybil log: {sybil_log_path}")
            if not os.path.exists(sybil_log_path):
                sybil_log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "..", "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json")
                log_message(f"Trying yet another alternative path for sybil log: {sybil_log_path}")
                if not os.path.exists(sybil_log_path):
                    log_message("Sybil attack log file sybil_attack_log.json not found. Proceeding without it.")
                else:
                    with open(sybil_log_path, "r", encoding='utf-8') as f:
                        sybil_log_data = json.load(f)
                        log_message("Successfully loaded sybil_attack_log.json.")
            else:
                with open(sybil_log_path, "r", encoding='utf-8') as f:
                    sybil_log_data = json.load(f)
                    log_message("Successfully loaded sybil_attack_log.json.")
        else:
            with open(sybil_log_path, "r", encoding='utf-8') as f:
                sybil_log_data = json.load(f)
                log_message("Successfully loaded sybil_attack_log.json.")
    except Exception as e:
        log_message(f"Error reading sybil_attack_log.json: {e}. Proceeding without it.")
        sybil_log_data = None

    # --- Extract node addresses and their activities ---
    node_data_for_fl = [] # Renamed to avoid confusion with 'node_data' from context if used directly
    all_node_addresses = set() # Using a set to store unique addresses

    # Helper to add node if address is present
    def add_node_if_present(address, role_name, activities_list):
        if address and address not in all_node_addresses:
            node_data_for_fl.append({
                "address": address,
                "role": role_name,
                "activities": activities_list # Activities will be populated by data prep script
            })
            all_node_addresses.add(address)
            log_message(f"Added {role_name} ({address}) to FL client list.")

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
        add_node_if_present(context.get(context_key), role_name, [])

    # Extract Sybil node addresses from sybil_attack_log.json
    if sybil_log_data and "sybilNodes" in sybil_log_data:
        for sybil_node_info in sybil_log_data["sybilNodes"]:
            sybil_address = sybil_node_info.get("address")
            # Role can be generic or from log if available, e.g., sybil_node_info.get("role", "SybilNode")
            add_node_if_present(sybil_address, f"SybilNode_{sybil_node_info.get('id', '')}", [])
    
    # Extract Bribed node addresses from sybil_attack_log.json (Scenario D)
    if sybil_log_data and "scenarioD" in sybil_log_data and "bribedNodes" in sybil_log_data["scenarioD"]:
        for bribed_node_info in sybil_log_data["scenarioD"]["bribedNodes"]:
            bribed_address = bribed_node_info.get("address")
            bribed_role = bribed_node_info.get("role", "BribedNode") # Get role from log if available
            add_node_if_present(bribed_address, f"Bribed_{bribed_role}", [])

    if not node_data_for_fl:
        log_message("No node data found in context or sybil log. Cannot proceed with FL training for node behavior.")
        # Optionally, create placeholder data if essential for testing structure
        # For now, we exit if no real participants are found.
        return False
    
    log_message(f"Total unique FL clients identified: {len(node_data_for_fl)}")

    # --- Run the Federated Learning model ---
    try:
        num_clients_to_use = len(node_data_for_fl)
        log_message(f"Running Node Behavior Timeseries with {num_clients_to_use} FL clients")
        
        log_message("Preparing federated data...")
        # Pass node_data_for_fl (which now just contains address and role) and sybil_log_data
        federated_data = make_federated_data_p3_timeseries_real(
            clients_info=node_data_for_fl, # List of dicts with {'address': ..., 'role': ...}
            sybil_attack_log=sybil_log_data
        )
        
        log_message(f"Successfully prepared federated data for {len(federated_data)} clients")
        
        # --- Save results ---
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "Node Behavior Timeseries",
            "num_clients_available": len(node_data_for_fl),
            "num_datasets_prepared": len(federated_data),
            "clients_info": node_data_for_fl,
            "sybil_log_loaded": sybil_log_data is not None
        }
        
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "node_behavior_timeseries_results.json")
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        log_message(f"Results saved to {results_path}")
        return True

    except Exception as e:
        log_message(f"Error during Node Behavior Timeseries FL model execution: {e}")
        import traceback
        log_message(traceback.format_exc()) # Log full traceback
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()
    log_message(f"Node Behavior Timeseries FL script finished in {end_time - start_time:.2f} seconds. Success: {success}")
