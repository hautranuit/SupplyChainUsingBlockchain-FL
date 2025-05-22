#!/usr/bin/env python3
"""
Real Data Preparation for Batch Monitoring FL Model

This module prepares real blockchain data for the batch monitoring federated learning model.
It extracts features from blockchain data and labels nodes as normal or anomalous based on
batch processing behavior, attack logs, or heuristic rules.
"""

import tensorflow as tf
import numpy as np
import time
import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batch_monitoring.log'))
    ]
)
logger = logging.getLogger("batch_monitoring")

# Adjust path to import connectors
current_dir = os.path.dirname(os.path.abspath(__file__))
fl_model_dir = os.path.dirname(current_dir)
sys.path.append(fl_model_dir)

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF, consistent with model_definition.py for Phase 2
NUM_BATCH_MONITORING_FEATURES = 6
ELEMENT_SPEC_BATCH_MONITORING = (
    tf.TensorSpec(shape=(NUM_BATCH_MONITORING_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for normal, 1 for anomalous
)

# Default RPC URL if not provided in environment
RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

# Heuristic thresholds for Batch Monitoring labeling (can be tuned)
PROPOSER_MIN_SUCCESS_RATE = 0.6 # If a proposer's success rate is below this, potentially anomalous
PROPOSER_MAX_FAILED_BATCHES_ABSOLUTE = 5 # If a proposer has more than this many failed batches
VALIDATOR_MIN_AGREEMENT_RATE = 0.7 # If a validator's agreement with final outcome is below this
VALIDATOR_LOW_PARTICIPATION_THRESHOLD = 0.1 # If validator participates in <10% of batches they were selected for

# Cache for batch events to avoid repeated blockchain queries
EVENT_CACHE = {
    "BatchProposed": None,
    "BatchValidated": None,
    "BatchCommitted": None,
    "BatchStateChanged": None, # Added for counterfeit detection
    "processed": False
}

def fetch_and_cache_batch_events(bc_connector: BlockchainConnector) -> None:
    """
    Fetch and cache batch processing events from the blockchain
    
    Args:
        bc_connector: Blockchain connector instance
    """
    if EVENT_CACHE["processed"]:
        return
    
    logger.info("Fetching and caching batch processing events...")
    try:
        EVENT_CACHE["BatchProposed"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="BatchProposed", from_block=0
        )
        EVENT_CACHE["BatchValidated"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="BatchValidated", from_block=0
        )
        EVENT_CACHE["BatchCommitted"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="BatchCommitted", from_block=0
        )
        
        # Try to fetch BatchStateChanged events from BatchProcessing contract
        try:
            EVENT_CACHE["BatchStateChanged"] = bc_connector.get_events(
                contract_name="BatchProcessing", event_name="BatchStateChanged", from_block=0
            )
        except Exception as e:
            logger.warning(f"Could not fetch BatchStateChanged events: {e}. This is not critical.")
            EVENT_CACHE["BatchStateChanged"] = []
        
        EVENT_CACHE["processed"] = True
        logger.info(f"Fetched {len(EVENT_CACHE['BatchProposed'])} BatchProposed events.")
        logger.info(f"Fetched {len(EVENT_CACHE['BatchValidated'])} BatchValidated events.")
        logger.info(f"Fetched {len(EVENT_CACHE['BatchCommitted'])} BatchCommitted events.")
        logger.info(f"Fetched {len(EVENT_CACHE['BatchStateChanged'])} BatchStateChanged events.")
    except Exception as e:
        logger.error(f"Error fetching batch events: {e}")
        EVENT_CACHE["processed"] = False
        for key in EVENT_CACHE:
            if key != "processed":
                EVENT_CACHE[key] = []

def is_known_sybil_or_bribed_node(node_address: str, sybil_attack_log: Optional[Dict[str, Any]]) -> bool:
    """
    Check if a node is a known Sybil or bribed node from the attack log
    
    Args:
        node_address: Address of the node to check
        sybil_attack_log: Attack log data
        
    Returns:
        True if the node is a known Sybil or bribed node, False otherwise
    """
    if not sybil_attack_log:
        return False
    
    # Check if node is in sybilNodes list
    if "sybilNodes" in sybil_attack_log:
        for node_info in sybil_attack_log["sybilNodes"]:
            if node_info.get("address") == node_address:
                return True
    
    # Check if node is in bribedNodes list in scenarioD
    if "scenarioD" in sybil_attack_log and "bribedNodes" in sybil_attack_log["scenarioD"]:
        for node_info in sybil_attack_log["scenarioD"]["bribedNodes"]:
            if node_info.get("address") == node_address:
                return True
    
    # Check flIntegrationMetadata if available
    if "flIntegrationMetadata" in sybil_attack_log:
        metadata = sybil_attack_log["flIntegrationMetadata"]
        
        if "sybilNodeAddresses" in metadata and node_address in metadata["sybilNodeAddresses"]:
            return True
            
        if "bribedNodeAddresses" in metadata and node_address in metadata["bribedNodeAddresses"]:
            return True
    
    return False

def generate_features_and_label_for_batch_monitoring(
    node_address_checksum: str, 
    all_events: Dict[str, List[Dict[str, Any]]], 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate features and label for batch monitoring for a single node
    
    Args:
        node_address_checksum: Checksummed address of the node
        all_events: Dictionary of all relevant blockchain events
        sybil_attack_log: Optional attack log data
        
    Returns:
        Tuple of (features array, label array)
    """
    features = np.zeros(NUM_BATCH_MONITORING_FEATURES, dtype=np.float32)
    anomaly_score = 0
    
    # Check if the node is a known Sybil or bribed node
    is_known_malicious = is_known_sybil_or_bribed_node(node_address_checksum, sybil_attack_log)
    
    # Proposer features and labeling
    proposed_by_node = [e for e in all_events.get("BatchProposed", []) if e["args"]["proposer"] == node_address_checksum]
    num_batches_proposed = len(proposed_by_node)
    num_proposed_committed_successfully = 0
    num_proposed_failed = 0
    total_validators_in_proposed_batches = 0

    if num_batches_proposed > 0:
        features[0] = float(num_batches_proposed) # F0: Number of batches proposed
        for prop_event in proposed_by_node:
            batch_id = prop_event["args"]["batchId"]
            total_validators_in_proposed_batches += len(prop_event["args"]["selectedValidators"])
            commit_event = next((ce for ce in all_events.get("BatchCommitted", []) if ce["args"]["batchId"] == batch_id), None)
            if commit_event and commit_event["args"]["success"]:
                num_proposed_committed_successfully += 1
            elif commit_event and not commit_event["args"]["success"]:
                num_proposed_failed +=1
            # If no commit event, it's also a form of failure or incompletion
            elif not commit_event:
                num_proposed_failed +=1 
        
        success_rate = (num_proposed_committed_successfully / num_batches_proposed) if num_batches_proposed > 0 else 0.0
        features[1] = float(success_rate) # F1: Success rate of proposed batches
        features[4] = float(total_validators_in_proposed_batches / num_batches_proposed) if num_batches_proposed > 0 else 0.0 # F4: Avg validators per proposed batch

        if success_rate < PROPOSER_MIN_SUCCESS_RATE and num_batches_proposed > 2: # Don't penalize too early
            anomaly_score += 2
            logger.info(f"Node {node_address_checksum} has low batch proposal success rate: {success_rate:.2f}")
        if num_proposed_failed > PROPOSER_MAX_FAILED_BATCHES_ABSOLUTE:
            anomaly_score += 1
            logger.info(f"Node {node_address_checksum} has high absolute number of failed batches: {num_proposed_failed}")

    # Validator features and labeling
    validated_by_node = [e for e in all_events.get("BatchValidated", []) if e["args"]["validator"] == node_address_checksum]
    num_batches_validated_participation = len(validated_by_node)
    num_agreements_with_outcome = 0
    num_approve_votes_by_node = 0
    num_collusive_votes = 0 # New feature for Sybils

    if num_batches_validated_participation > 0:
        features[2] = float(num_batches_validated_participation) # F2: Number of batches validated
        for val_event in validated_by_node:
            batch_id = val_event["args"]["batchId"]
            my_vote_approved = val_event["args"]["approve"]
            if my_vote_approved:
                num_approve_votes_by_node += 1
            
            commit_event = next((ce for ce in all_events.get("BatchCommitted", []) if ce["args"]["batchId"] == batch_id), None)
            if commit_event:
                batch_outcome_success = commit_event["args"]["success"]
                if my_vote_approved == batch_outcome_success:
                    num_agreements_with_outcome += 1
        
        agreement_rate = (num_agreements_with_outcome / num_batches_validated_participation) if num_batches_validated_participation > 0 else 0.0
        features[3] = float(agreement_rate) # F3: Agreement rate with final batch outcome
        features[5] = float(num_approve_votes_by_node / num_batches_validated_participation) if num_batches_validated_participation > 0 else 0.0 # F5: Approval voting rate

        if agreement_rate < VALIDATOR_MIN_AGREEMENT_RATE and num_batches_validated_participation > 2:
            anomaly_score += 2
            logger.info(f"Node {node_address_checksum} has low agreement rate with batch outcomes: {agreement_rate:.2f}")
        
        # Rough participation rate check
        total_batches_in_system = len(all_events.get("BatchProposed", []))
        if total_batches_in_system > 0 and (num_batches_validated_participation / total_batches_in_system) < VALIDATOR_LOW_PARTICIPATION_THRESHOLD:
            if num_batches_proposed == 0: # Only penalize if not also a proposer (proposers might not validate much)
                anomaly_score += 1
                logger.info(f"Node {node_address_checksum} has low participation rate in batch validation: {(num_batches_validated_participation / total_batches_in_system):.2f}")

    # Check for Scenario B (Collusive Voting) and Scenario C (Counterfeit Batch) from sybil_attack_log
    if sybil_attack_log:
        # Check for collusive voting in Scenario B
        if "scenarioB" in sybil_attack_log and "actions" in sybil_attack_log["scenarioB"]:
            for action in sybil_attack_log["scenarioB"]["actions"]:
                if action.get("type") == "collusiveVoting" and action.get("sybilNode") == node_address_checksum:
                    batch_id = action.get("batchId")
                    if batch_id:
                        # Check if this node actually voted on this batch according to blockchain events
                        corresponding_vote_event = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id), None)
                        if corresponding_vote_event:
                            num_collusive_votes += 1
                            anomaly_score += 2 # Strong indicator
                            logger.info(f"Node {node_address_checksum} detected in collusive voting (Scenario B) for batch {batch_id}")
        
        # Check for counterfeit batch in Scenario C
        if "scenarioC" in sybil_attack_log:
            # Check if node proposed counterfeit batch
            if "actions" in sybil_attack_log["scenarioC"]:
                for action in sybil_attack_log["scenarioC"]["actions"]:
                    if action.get("type") == "counterfeitBatchProposal" and action.get("proposerNode") == node_address_checksum:
                        anomaly_score += 3 # Very strong indicator
                        logger.info(f"Node {node_address_checksum} detected proposing counterfeit batch (Scenario C)")
                    
                    # Check if node approved counterfeit batch
                    if action.get("type") == "counterfeitBatchApproval" and action.get("sybilNode") == node_address_checksum:
                        batch_id = action.get("batchId")
                        if batch_id:
                            corresponding_vote_event = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id), None)
                            if corresponding_vote_event and corresponding_vote_event["args"]["approve"]:
                                anomaly_score += 2 # Strong indicator for approving
                                logger.info(f"Node {node_address_checksum} detected approving counterfeit batch (Scenario C) for batch {batch_id}")
            
            # Also check details section for counterfeit batch info
            if "details" in sybil_attack_log["scenarioC"] and "counterfeitBatchId" in sybil_attack_log["scenarioC"]["details"]:
                counterfeit_batch_id = sybil_attack_log["scenarioC"]["details"]["counterfeitBatchId"]
                
                # Check if node proposed this batch
                if any(e["args"]["batchId"] == counterfeit_batch_id for e in proposed_by_node):
                    anomaly_score += 3
                    logger.info(f"Node {node_address_checksum} proposed the known counterfeit batch {counterfeit_batch_id}")
                
                # Check if node approved this batch
                counterfeit_vote = next((ve for ve in validated_by_node if ve["args"]["batchId"] == counterfeit_batch_id and ve["args"]["approve"]), None)
                if counterfeit_vote:
                    anomaly_score += 2
                    logger.info(f"Node {node_address_checksum} approved the known counterfeit batch {counterfeit_batch_id}")
    
    # Check for attack details in flIntegrationMetadata
    if sybil_attack_log and "flIntegrationMetadata" in sybil_attack_log:
        metadata = sybil_attack_log["flIntegrationMetadata"]
        
        # If node is in sybilNodeAddresses, increase anomaly score
        if "sybilNodeAddresses" in metadata and node_address_checksum in metadata["sybilNodeAddresses"]:
            anomaly_score += 2
            logger.info(f"Node {node_address_checksum} is a known Sybil node from flIntegrationMetadata")
        
        # If node is in bribedNodeAddresses, increase anomaly score
        if "bribedNodeAddresses" in metadata and node_address_checksum in metadata["bribedNodeAddresses"]:
            anomaly_score += 1
            logger.info(f"Node {node_address_checksum} is a known bribed node from flIntegrationMetadata")
    
    # If node is neither proposer nor validator in any significant way
    if num_batches_proposed == 0 and num_batches_validated_participation == 0 and len(all_events.get("BatchProposed",[])) > 5:
        # System has batches but this node is inactive
        anomaly_score += 1
        logger.info(f"Node {node_address_checksum} is inactive in batch processing despite system activity")

    # Final labeling
    label_val = 1 if anomaly_score >= 2 or is_known_malicious else 0
    label = np.array([label_val], dtype=np.int32)
    
    if label_val == 1:
        logger.info(f"Node {node_address_checksum} labeled ANOMALOUS for Batch Monitoring (score: {anomaly_score}). " +
                   f"Features: [{features[0]:.0f},{features[1]:.2f},{features[2]:.0f},{features[3]:.2f},{features[4]:.1f},{features[5]:.2f}], " +
                   f"Known malicious: {is_known_malicious}")
    
    return features, label

def load_real_data_for_fl_client(
    client_id: str, 
    target_node_addresses: List[str], 
    bc_connector: BlockchainConnector, 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> tf.data.Dataset:
    """
    Load real data for a federated learning client
    
    Args:
        client_id: Identifier for the client
        target_node_addresses: List of node addresses to process
        bc_connector: Blockchain connector instance
        sybil_attack_log: Optional attack log data
        
    Returns:
        TensorFlow dataset for the client
    """
    logger.info(f"FL Client {client_id}: Loading real data for {len(target_node_addresses)} nodes")
    client_features_list = []
    client_labels_list = []

    # Ensure batch events are cached
    fetch_and_cache_batch_events(bc_connector)

    for node_addr in target_node_addresses:
        try:
            # Convert to checksum address
            node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr)
            logger.info(f"Client {client_id}: Processing node {node_addr_checksum}")
            
            # Generate features and label
            features, label = generate_features_and_label_for_batch_monitoring(
                node_addr_checksum, EVENT_CACHE, sybil_attack_log
            )
            
            client_features_list.append(features)
            client_labels_list.append(label)
        except Exception as e:
            logger.error(f"Error processing node {node_addr} for client {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not client_features_list:
        logger.warning(f"No valid data for client {client_id}. Returning empty dataset.")
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    logger.info(f"Client {client_id}: Created dataset with {len(client_features_list)} samples")
    return dataset.batch(1)

def find_sybil_attack_log() -> Optional[Dict[str, Any]]:
    """
    Find and load the sybil_attack_log.json file
    
    Returns:
        Dictionary containing sybil attack log data, or None if not found
    """
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "sybil_attack_log.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "sybil_attack_log.json")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded sybil_attack_log.json from {path}")
                return data
            except Exception as e:
                logger.error(f"Error reading sybil_attack_log.json from {path}: {e}")
    
    logger.warning("sybil_attack_log.json not found in any expected location")
    return None

def make_federated_data_batch_monitoring_real(
    all_node_addresses: List[str], 
    num_fl_clients: int = 3, 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> List[tf.data.Dataset]:
    """
    Create federated datasets for batch monitoring using real blockchain data
    
    Args:
        all_node_addresses: List of all node addresses to process
        num_fl_clients: Number of federated learning clients
        sybil_attack_log: Optional attack log data
        
    Returns:
        List of TensorFlow datasets, one for each client
    """
    logger.info(f"Creating federated data for batch monitoring with {num_fl_clients} clients")
    
    # Try to load sybil_attack_log if not provided
    if sybil_attack_log is None:
        sybil_attack_log = find_sybil_attack_log()
    
    # Initialize BlockchainConnector
    # Use the RPC_URL_OVERRIDE if provided, otherwise it will use default from .env or hardcoded
    rpc_url = RPC_URL_OVERRIDE if RPC_URL_OVERRIDE else None
    bc_connector = BlockchainConnector(rpc_url=rpc_url)

    if not bc_connector.web3 or not bc_connector.contract:
        logger.error("BlockchainConnector not properly initialized. Check RPC URL and contract loading.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if "SupplyChainNFT" not in bc_connector.contracts:
        logger.error("SupplyChainNFT contract not loaded. Cannot proceed with real data.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if not all_node_addresses:
        logger.warning("No target node addresses provided. Cannot proceed with real data.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    # Ensure batch events are cached
    fetch_and_cache_batch_events(bc_connector)

    # Distribute all_node_addresses among clients
    nodes_per_client = len(all_node_addresses) // num_fl_clients
    extra_nodes = len(all_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"batch_monitoring_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(all_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(all_node_addresses))
            client_node_list = all_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list: 
            logger.warning(f"No nodes assigned to client {client_id}. Creating empty dataset.")
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client(client_id, client_node_list, bc_connector, sybil_attack_log=sybil_attack_log)
        client_datasets.append(client_ds)
    
    logger.info(f"Created {len(client_datasets)} federated datasets for batch monitoring")
    return client_datasets

if __name__ == '__main__':
    logger.info("Testing Real Data Preparation for Batch Monitoring with Attack Log Integration...")
    
    # Find and load sybil_attack_log.json
    sybil_log_for_test = find_sybil_attack_log()
    
    if not sybil_log_for_test:
        # Create dummy log for testing
        logger.info("Creating dummy sybil_attack_log.json for testing")
        sybil_log_for_test = {
            "sybilNodes": [
                {
                    "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", 
                    "initialization": [
                        {"action": "adminUpdateReputation", "newReputation": 10, "timestamp": "2024-01-01T10:00:00.000Z"},
                        {"action": "setNodeType", "nodeType": 1, "timestamp": "2024-01-01T10:00:05.000Z"}
                    ],
                    "activities": []
                }
            ],
            "scenarioB": {
                "actions": [
                    {
                        "type": "collusiveVoting",
                        "batchId": "1",
                        "vote": "Deny",
                        "sybilNode": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        "status": "success"
                    }
                ]
            },
            "scenarioC": {
                "details": {
                    "counterfeitBatchId": "2"
                },
                "actions": [
                    {
                        "type": "counterfeitBatchProposal",
                        "batchId": "2",
                        "proposerNode": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        "status": "success"
                    }
                ]
            },
            "flIntegrationMetadata": {
                "sybilNodeAddresses": ["0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"]
            }
        }
    
    # Test with a few sample addresses
    test_addresses = [
        "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
        "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
        "0x90F79bf6EB2c4f870365E785982E1f101E93b906"
    ]
    
    if sybil_log_for_test and "sybilNodes" in sybil_log_for_test:
        for node in sybil_log_for_test["sybilNodes"]:
            if "address" in node and node["address"] not in test_addresses:
                test_addresses.append(node["address"])
    
    logger.info(f"Testing with addresses: {test_addresses}")
    
    # Create federated datasets
    federated_datasets = make_federated_data_batch_monitoring_real(
        all_node_addresses=test_addresses,
        num_fl_clients=2,
        sybil_attack_log=sybil_log_for_test
    )
    
    logger.info(f"Created {len(federated_datasets)} federated datasets")
    for i, ds in enumerate(federated_datasets):
        for features, labels in ds.take(1):
            logger.info(f"Client {i} dataset sample - Features shape: {features.shape}, Labels shape: {labels.shape}")
