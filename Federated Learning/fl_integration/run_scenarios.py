#!/usr/bin/env python
"""
Scenario Runner for Federated Learning Integration.
This script runs both normal and attack scenarios and compares their results.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scenario_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scenario_runner")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run FL scenarios (normal and attack)')
    
    parser.add_argument('--output_dir', type=str, default='./scenario_results',
                        help='Directory for scenario results')
    
    parser.add_argument('--run_normal', action='store_true', default=True,
                        help='Run normal scenario (scripts 01-06 + FL)')
    
    parser.add_argument('--run_attack', action='store_true', default=True,
                        help='Run attack scenario (scripts 01-06 + script 07 + FL)')
    
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Compare results of both scenarios')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def setup_directories(args):
    """Set up output directories."""
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create scenario directories
    normal_dir = os.path.join(args.output_dir, 'normal_scenario')
    attack_dir = os.path.join(args.output_dir, 'attack_scenario')
    comparison_dir = os.path.join(args.output_dir, 'comparison')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    return {
        'normal_dir': normal_dir,
        'attack_dir': attack_dir,
        'comparison_dir': comparison_dir
    }

def run_normal_scenario(output_dir, verbose=False):
    """Run normal scenario (scripts 01-06 + FL)."""
    logger.info("Starting NORMAL scenario...")
    
    try:
        # Run the federated learning script with normal data
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_federated_learning.py"),
            "--mode", "full",
            "--output_dir", output_dir
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Log the output
        with open(os.path.join(output_dir, "normal_scenario_output.log"), "w") as f:
            f.write(result.stdout)
        
        if verbose:
            logger.info(result.stdout)
        
        logger.info("NORMAL scenario completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running NORMAL scenario: {str(e)}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running NORMAL scenario: {str(e)}")
        return False

def run_attack_scenario(output_dir, verbose=False):
    """Run attack scenario (scripts 01-06 + script 07 + FL)."""
    logger.info("Starting ATTACK scenario...")
    
    try:
        # First, run script 07 to generate attack data if it exists
        script07_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "SupplyChain_dapp/scripts/lifecycle_demo/07_simulate_attack.js"
        )
        
        if os.path.exists(script07_path):
            logger.info("Running script 07 to generate attack data...")
            
            try:
                # Change to the directory containing the script
                original_dir = os.getcwd()
                script_dir = os.path.dirname(script07_path)
                os.chdir(script_dir)
                
                # Run the script using Node.js
                node_cmd = ["node", os.path.basename(script07_path)]
                
                node_result = subprocess.run(
                    node_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Change back to the original directory
                os.chdir(original_dir)
                
                # Log the output
                with open(os.path.join(output_dir, "script07_output.log"), "w") as f:
                    f.write(node_result.stdout)
                    f.write("\n\nSTDERR:\n")
                    f.write(node_result.stderr)
                
                if node_result.returncode != 0:
                    logger.warning(f"Script 07 failed with code {node_result.returncode}")
                    logger.warning(node_result.stderr)
                else:
                    logger.info("Script 07 completed successfully")
                    
                if verbose:
                    logger.info(node_result.stdout)
                
            except Exception as e:
                logger.warning(f"Error running script 07: {str(e)}")
                logger.warning("Continuing with attack scenario anyway...")
        else:
            logger.warning(f"Script 07 not found at {script07_path}")
            logger.warning("Simulating attack scenario without running script 07")
            
            # Create a mock attack_context.json file for testing
            mock_attack_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mock_attack_path = os.path.join(mock_attack_dir, "attack_context.json")
            
            if not os.path.exists(mock_attack_path):
                logger.info("Creating mock attack data for testing...")
                
                # Create a basic mock attack data structure
                mock_attack_data = {
                    "nodes": {
                        "sybil_node_1": {
                            "id": "sybil_node_1",
                            "address": "0x1234567890abcdef1234567890abcdef12345678",
                            "type": "malicious",
                            "status": "active"
                        },
                        "sybil_node_2": {
                            "id": "sybil_node_2",
                            "address": "0xabcdef1234567890abcdef1234567890abcdef12",
                            "type": "malicious",
                            "status": "active"
                        }
                    },
                    "batches": {
                        "tampered_batch_1": {
                            "id": "tampered_batch_1",
                            "status": "compromised",
                            "data": "manipulated_data"
                        }
                    }
                }
                
                with open(mock_attack_path, "w") as f:
                    json.dump(mock_attack_data, f, indent=2)
                logger.info(f"Mock attack data created at {mock_attack_path}")
        
        # Now run the federated learning script with attack data
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_federated_learning.py"),
            "--mode", "full",
            "--output_dir", output_dir
        ]
        
        if verbose:
            cmd.append("--verbose")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Log the output
        with open(os.path.join(output_dir, "attack_scenario_output.log"), "w") as f:
            f.write(result.stdout)
        
        if verbose:
            logger.info(result.stdout)
        
        logger.info("ATTACK scenario completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ATTACK scenario: {str(e)}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running ATTACK scenario: {str(e)}")
        return False

def compare_results(normal_dir, attack_dir, comparison_dir):
    """Compare results of both scenarios."""
    logger.info("Comparing scenario results...")
    
    try:
        # Load detection results from both scenarios
        normal_results_path = os.path.join(normal_dir, "results", "detection_results.json")
        attack_results_path = os.path.join(attack_dir, "results", "detection_results.json")
        
        if not os.path.exists(normal_results_path):
            logger.error(f"Normal scenario results not found at {normal_results_path}")
            return False
        
        if not os.path.exists(attack_results_path):
            logger.error(f"Attack scenario results not found at {attack_results_path}")
            return False
        
        with open(normal_results_path, "r") as f:
            normal_results = json.load(f)
        
        with open(attack_results_path, "r") as f:
            attack_results = json.load(f)
        
        # Create a comparison report
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "normal_scenario": {
                "attack_detected": normal_results.get("attack_detected", False),
                "confidence": normal_results.get("confidence", 0.0)
            },
            "attack_scenario": {
                "attack_detected": attack_results.get("attack_detected", False),
                "confidence": attack_results.get("confidence", 0.0)
            },
            "comparison": {
                "detection_difference": (
                    attack_results.get("confidence", 0.0) - 
                    normal_results.get("confidence", 0.0)
                ),
                "correct_detection": (
                    not normal_results.get("attack_detected", False) and 
                    attack_results.get("attack_detected", False)
                )
            }
        }
        
        # Add detection details if available
        if "details" in normal_results:
            comparison["normal_scenario"]["details"] = {
                "sybil_nodes_count": len(normal_results["details"].get("sybil_nodes", [])),
                "suspicious_batches_count": len(normal_results["details"].get("suspicious_batches", [])),
                "bribery_attacks_count": len(normal_results["details"].get("bribery_attacks", []))
            }
        
        if "details" in attack_results:
            comparison["attack_scenario"]["details"] = {
                "sybil_nodes_count": len(attack_results["details"].get("sybil_nodes", [])),
                "suspicious_batches_count": len(attack_results["details"].get("suspicious_batches", [])),
                "bribery_attacks_count": len(attack_results["details"].get("bribery_attacks", []))
            }
        
        # Add detailed reports if available
        if "detailed_report" in normal_results:
            comparison["normal_scenario"]["threat_level"] = normal_results["detailed_report"].get("threat_level", "N/A")
            comparison["normal_scenario"]["summary"] = normal_results["detailed_report"].get("summary", "N/A")
        
        if "detailed_report" in attack_results:
            comparison["attack_scenario"]["threat_level"] = attack_results["detailed_report"].get("threat_level", "N/A")
            comparison["attack_scenario"]["summary"] = attack_results["detailed_report"].get("summary", "N/A")
        
        # Save comparison report
        comparison_path = os.path.join(comparison_dir, "scenario_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Print comparison summary
        print("\n" + "="*70)
        print("SCENARIO COMPARISON RESULTS")
        print("="*70)
        print(f"Normal scenario detection: {'YES' if normal_results.get('attack_detected', False) else 'NO'}")
        print(f"Attack scenario detection: {'YES' if attack_results.get('attack_detected', False) else 'NO'}")
        print(f"Detection confidence difference: {comparison['comparison']['detection_difference']:.4f}")
        print(f"Correct detection in both scenarios: {'YES' if comparison['comparison']['correct_detection'] else 'NO'}")
        print("\nDetailed summary:")
        print(f"- Normal scenario: {comparison['normal_scenario'].get('summary', 'N/A')}")
        print(f"- Attack scenario: {comparison['attack_scenario'].get('summary', 'N/A')}")
        print("\nComparison report saved to:")
        print(f"- {comparison_path}")
        print("="*70 + "\n")
        
        logger.info("Scenario comparison completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error comparing scenario results: {str(e)}")
        return False

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up directories
    dirs = setup_directories(args)
    
    # Initialize results
    normal_success = False
    attack_success = False
    
    # Run normal scenario if requested
    if args.run_normal:
        normal_success = run_normal_scenario(dirs['normal_dir'], args.verbose)
    
    # Run attack scenario if requested
    if args.run_attack:
        attack_success = run_attack_scenario(dirs['attack_dir'], args.verbose)
    
    # Compare results if requested and both scenarios ran successfully
    if args.compare and normal_success and attack_success:
        compare_results(dirs['normal_dir'], dirs['attack_dir'], dirs['comparison_dir'])
    
    # Print final status
    print("\n" + "="*70)
    print("SCENARIO RUNNER SUMMARY")
    print("="*70)
    print(f"Normal scenario: {'SUCCESS' if normal_success else 'FAILED'}")
    print(f"Attack scenario: {'SUCCESS' if attack_success else 'FAILED'}")
    print(f"Comparison: {'COMPLETED' if args.compare and normal_success and attack_success else 'NOT AVAILABLE'}")
    print("\nResults saved to:")
    print(f"- Normal scenario: {dirs['normal_dir']}")
    print(f"- Attack scenario: {dirs['attack_dir']}")
    print(f"- Comparison: {dirs['comparison_dir']}")
    print("="*70 + "\n")
    
    if normal_success and attack_success:
        logger.info("All scenarios completed successfully")
        return 0
    else:
        logger.error("Some scenarios failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
