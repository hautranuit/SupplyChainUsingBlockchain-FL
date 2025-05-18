#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fl_integration_run.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def check_script_existence():
    """Check if all required FL scripts exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    required_scripts = [
        "run_sybil_detection.py",
        "run_batch_monitoring.py",
        "run_node_behavior_timeseries.py",
        "run_arbitrator_bias.py",
        "run_dispute_risk.py"
    ]
    
    missing_scripts = []
    for script in required_scripts:
        script_path = os.path.join(script_dir, script)
        if not os.path.exists(script_path):
            missing_scripts.append(script)
    
    return missing_scripts

def check_context_file():
    """Check if demo_context.json exists and is valid"""
    # First try the standard path (relative to project root)
    context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                               "SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json")
    
    if not os.path.exists(context_path):
        # Try alternative path (in Federated Learning directory)
        context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "demo_context.json")
        
        if not os.path.exists(context_path):
            return False, "Context file not found"
    
    try:
        with open(context_path, "r", encoding='utf-8') as f:
            context = json.load(f)
        return True, f"Context file valid with {len(context.keys())} keys"
    except Exception as e:
        return False, f"Error parsing context file: {e}"

def check_master_script():
    """Check if master orchestration script exists"""
    master_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "run_integrated_system.js")
    
    if not os.path.exists(master_script_path):
        return False, "Master orchestration script not found"
    
    return True, "Master orchestration script found"

def check_results_directory():
    """Check if results directory exists, create if not"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir, exist_ok=True)
            return True, f"Results directory created at {results_dir}"
        except Exception as e:
            return False, f"Error creating results directory: {e}"
    
    return True, f"Results directory exists at {results_dir}"

def main():
    log_message("=== Starting Integration Validation ===")
    
    # Check if all required FL scripts exist
    missing_scripts = check_script_existence()
    if missing_scripts:
        log_message(f"WARNING: The following FL scripts are missing: {', '.join(missing_scripts)}")
    else:
        log_message("All required FL scripts exist")
    
    # Check if context file exists and is valid
    context_valid, context_message = check_context_file()
    log_message(f"Context file check: {context_message}")
    
    # Check if master orchestration script exists
    master_valid, master_message = check_master_script()
    log_message(f"Master script check: {master_message}")
    
    # Check if results directory exists
    results_valid, results_message = check_results_directory()
    log_message(f"Results directory check: {results_message}")
    
    # Validate data flow between scripts
    log_message("Validating data flow between scripts...")
    
    # Create a simple test context if none exists
    if not context_valid:
        log_message("Creating a test context file for validation...")
        test_context = {
            "contractAddress": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
            "deployerAddress": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "manufacturerAddress": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
            "transporter1Address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
            "transporter2Address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906",
            "transporter3Address": "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65",
            "retailerAddress": "0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc",
            "buyer1Address": "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
            "buyer2Address": "0x14dC79964da2C08b23698B3D3cc7Ca32193d9955",
            "arbitratorAddress": "0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f",
            "productDetails": [
                {
                    "tokenId": "1",
                    "uniqueProductID": "DEMO_PROD_001",
                    "batchNumber": "B_ALPHA_001",
                    "manufacturingDate": "2025-05-10",
                    "expirationDate": "2027-05-10",
                    "productType": "Electronics - HighEnd Laptop",
                    "manufacturerID": "MANU_ACME_CORP",
                    "currentOwnerAddress": "0x976EA74026E726554dB657fA54763abd0C3a0aa9",
                    "transportStatus": "Completed"
                }
            ]
        }
        
        test_context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_context.json")
        with open(test_context_path, "w", encoding='utf-8') as f:
            json.dump(test_context, f, indent=2)
        
        log_message(f"Test context created at {test_context_path}")
    
    # Test individual FL scripts with minimal execution
    log_message("Testing individual FL scripts...")
    
    script_results = {}
    for script_name in ["run_sybil_detection.py", "run_batch_monitoring.py", "run_node_behavior_timeseries.py", 
                        "run_arbitrator_bias.py", "run_dispute_risk.py"]:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        
        if os.path.exists(script_path):
            log_message(f"Validating script: {script_name}")
            
            # Check imports and basic structure
            try:
                with open(script_path, "r", encoding='utf-8') as f:
                    script_content = f.read()
                
                # Check for essential components
                has_make_federated_data = "make_federated_data" in script_content
                has_log_message = "log_message" in script_content
                has_context_reading = "context_path" in script_content and "json.load" in script_content
                has_results_saving = "results_path" in script_content and "json.dump" in script_content
                
                script_results[script_name] = {
                    "exists": True,
                    "has_make_federated_data": has_make_federated_data,
                    "has_log_message": has_log_message,
                    "has_context_reading": has_context_reading,
                    "has_results_saving": has_results_saving,
                    "validation_passed": has_make_federated_data and has_log_message and has_context_reading and has_results_saving
                }
                
                log_message(f"  - make_federated_data function: {'Found' if has_make_federated_data else 'Missing'}")
                log_message(f"  - log_message function: {'Found' if has_log_message else 'Missing'}")
                log_message(f"  - Context reading logic: {'Found' if has_context_reading else 'Missing'}")
                log_message(f"  - Results saving logic: {'Found' if has_results_saving else 'Missing'}")
                
                if script_results[script_name]["validation_passed"]:
                    log_message(f"  [PASS] Script {script_name} passed validation")
                else:
                    log_message(f"  [FAIL] Script {script_name} failed validation")
            
            except Exception as e:
                log_message(f"  Error validating {script_name}: {e}")
                script_results[script_name] = {
                    "exists": True,
                    "error": str(e),
                    "validation_passed": False
                }
        else:
            log_message(f"Script {script_name} does not exist, skipping validation")
            script_results[script_name] = {
                "exists": False,
                "validation_passed": False
            }
    
    # Save validation results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "missing_scripts": missing_scripts,
        "context_file": {
            "valid": context_valid,
            "message": context_message
        },
        "master_script": {
            "valid": master_valid,
            "message": master_message
        },
        "results_directory": {
            "valid": results_valid,
            "message": results_message
        },
        "script_validation": script_results,
        "overall_status": "PASS" if not missing_scripts and context_valid and master_valid and results_valid else "FAIL"
    }
    
    validation_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "integration_validation_results.json")
    with open(validation_results_path, "w", encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2)
    
    log_message(f"Validation results saved to {validation_results_path}")
    
    # Overall assessment
    if validation_results["overall_status"] == "PASS":
        log_message("[PASS] Integration validation PASSED. All components are properly configured for workflow execution.")
    else:
        log_message("[FAIL] Integration validation FAILED. Please address the issues mentioned above.")
    
    log_message("=== Integration Validation Complete ===")
    return validation_results["overall_status"] == "PASS"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
