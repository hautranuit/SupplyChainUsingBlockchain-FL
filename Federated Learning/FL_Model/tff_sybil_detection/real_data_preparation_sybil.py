#!/usr/bin/env python3
"""
Real Data Preparation for Sybil Detection FL Model

This module prepares real blockchain data for the Sybil detection federated learning model.
It extracts features from blockchain data and labels nodes as Sybil or normal based on
attack logs or heuristic rules.
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
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sybil_detection.log'))
    ]
)
logger = logging.getLogger("sybil_detection")

# Adjust path to import connectors
current_dir = os.path.dirname(os.path.abspath(__file__))
fl_model_dir = os.path.dirname(current_dir)
sys.path.append(fl_model_dir)

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF, consistent with the Sybil detection model_definition.py
NUM_SYBIL_FEATURES = 5
ELEMENT_SPEC_SYBIL = (
    tf.TensorSpec(shape=(NUM_SYBIL_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)     # Labels (0 for normal, 1 for Sybil)
)

# Default RPC URL if not provided in environment
RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

# Heuristic thresholds for Sybil labeling (can be tuned)
SYBIL_MAX_AGE_SECONDS = 86400 * 7  # 7 days
SYBIL_MAX_REPUTATION = 30  # Reputation threshold
SYBIL_MAX_TX_COUNT = 5    # Transaction count threshold
SYBIL_MAX_COUNTERPARTIES = 2  # Counterparties threshold

def get_node_registration_timestamp(bc_connector: BlockchainConnector, node_address: str) -> int:
    """
    Get the timestamp when a node was first registered/verified on the blockchain
    
    Args:
        bc_connector: Blockchain connector instance
        node_address: Address of the node
        
    Returns:
        Unix timestamp of registration, or 0 if not found
    """
    try:
        checksum_node_address = bc_connector.w3.to_checksum_address(node_address)
        node_verified_events = bc_connector.get_events(
            contract_name="SupplyChainNFT",
            event_name="NodeVerified",
            argument_filters={"node": checksum_node_address},
            from_block=0
        )
        if node_verified_events:
            node_verified_events.sort(key=lambda x: x["blockNumber"])
            first_verification_event = node_verified_events[0]
            block_number = first_verification_event["blockNumber"]
            timestamp = bc_connector.w3.eth.get_block(block_number)["timestamp"]
            return timestamp
        return 0
    except Exception as e:
        logger.error(f"Error getting registration timestamp for {node_address}: {e}")
        return 0

def get_node_transaction_count_and_counterparties(bc_connector: BlockchainConnector, node_address: str) -> Tuple[int, int]:
    """
    Get the transaction count and number of distinct counterparties for a node
    
    Args:
        bc_connector: Blockchain connector instance
        node_address: Address of the node
        
    Returns:
        Tuple of (transaction count, number of distinct counterparties)
    """
    transaction_count = 0
    counterparties = set()
    try:
        checksum_node_address = bc_connector.w3.to_checksum_address(node_address)
        event_types_and_roles = [
            ("ProductMinted", "owner"),
            ("InitialCIDStored", "actor"),
            ("DirectSaleAndTransferCompleted", "seller", "buyer"),
            ("PaymentAndTransferCompleted", "seller", "buyer"),
            ("DisputeInitiated", "initiator", "currentOwner"), # currentOwner is the other party
            ("BatchProposed", "proposer"),
            ("BatchValidated", "validator"),
            ("ArbitratorVoted", "voter")
        ]
        for event_info in event_types_and_roles:
            event_name = event_info[0]
            roles = event_info[1:]
            for role in roles:
                # Fetch events where the node is in this role
                filters = {role: checksum_node_address}
                events = bc_connector.get_events(
                    contract_name="SupplyChainNFT",
                    event_name=event_name,
                    argument_filters=filters,
                    from_block=0
                )
                transaction_count += len(events)
                # Identify counterparties
                for event_data in events:
                    for other_role in roles:
                        if other_role != role and other_role in event_data["args"]:
                            counterparty = event_data["args"][other_role]
                            if isinstance(counterparty, str) and bc_connector.w3.to_checksum_address(counterparty) != checksum_node_address:
                                counterparties.add(bc_connector.w3.to_checksum_address(counterparty))
        if checksum_node_address in counterparties:
            counterparties.remove(checksum_node_address)
        return transaction_count, len(counterparties)
    except Exception as e:
        logger.error(f"Error in get_node_transaction_count_and_counterparties for {node_address}: {e}")
        return 0, 0

def parse_iso_timestamp(timestamp_str: str) -> int:
    """
    Parse ISO format timestamp string to Unix timestamp
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        Unix timestamp (seconds since epoch)
    """
    try:
        # Handle potential 'Z' for UTC
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return int(datetime.fromisoformat(timestamp_str).timestamp())
    except Exception as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return 0

def generate_features_and_label_for_sybil(
    bc_connector: BlockchainConnector, 
    node_address: str, 
    sybil_attack_log: Optional[Dict[str, Any]] = None, 
    fl_model: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate features and label for Sybil detection for a single node
    
    Args:
        bc_connector: Blockchain connector instance
        node_address: Address of the node
        sybil_attack_log: Optional attack log data
        fl_model: Optional trained FL model for prediction
        
    Returns:
        Tuple of (features array, label array)
    """
    features = np.zeros(NUM_SYBIL_FEATURES, dtype=np.float32)
    current_timestamp = int(time.time())
    is_known_sybil = False
    sybil_node_details = None

    # Check if this node is a known Sybil from the log
    if sybil_attack_log and "sybilNodes" in sybil_attack_log:
        for node_info in sybil_attack_log["sybilNodes"]:
            if node_info.get("address") == node_address:
                is_known_sybil = True
                sybil_node_details = node_info
                logger.info(f"Node {node_address} identified as a known Sybil from sybil_attack_log.json.")
                break
    
    # Also check flIntegrationMetadata if available for Sybils
    if not is_known_sybil and sybil_attack_log and "flIntegrationMetadata" in sybil_attack_log:
        if "sybilNodeAddresses" in sybil_attack_log["flIntegrationMetadata"]:
            if node_address in sybil_attack_log["flIntegrationMetadata"]["sybilNodeAddresses"]:
                is_known_sybil = True
                logger.info(f"Node {node_address} identified as a known Sybil from flIntegrationMetadata.")

    # Feature 1: Age (can use initialization timestamp from log if available for Sybils)
    reg_timestamp = 0
    if is_known_sybil and sybil_node_details and "initialization" in sybil_node_details and sybil_node_details["initialization"]:
        # Find the earliest timestamp from initialization actions if multiple exist
        init_timestamps = [action.get("timestamp") for action in sybil_node_details["initialization"] if action.get("timestamp")]
        if init_timestamps:
            # Take the first one as the primary registration/initialization time for this Sybil node
            iso_ts_str = init_timestamps[0]
            reg_timestamp = parse_iso_timestamp(iso_ts_str)
            if reg_timestamp > 0:
                logger.info(f"Using initialization timestamp from log for Sybil {node_address}: {reg_timestamp}")
            else:
                reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)
        else:
            reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)
    else:
        reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)

    age_seconds = 0
    if reg_timestamp > 0 and reg_timestamp < current_timestamp:
        age_seconds = float(current_timestamp - reg_timestamp)
        features[0] = age_seconds
    else:
        features[0] = 0.0

    # Feature 2: Reputation (can use initial reputation from log if available for Sybils)
    reputation = 0
    if is_known_sybil and sybil_node_details and "initialization" in sybil_node_details:
        # Look for adminUpdateReputation in initialization actions
        rep_action = next((act for act in sybil_node_details["initialization"] if act.get("action") == "adminUpdateReputation"), None)
        if rep_action and "newReputation" in rep_action:
            reputation = float(rep_action["newReputation"])
            logger.info(f"Using initial reputation from log for Sybil {node_address}: {reputation}")
        else:
            reputation_bc = bc_connector.get_node_reputation(node_address)
            reputation = float(reputation_bc if reputation_bc is not None else 0)
    else:
        reputation_bc = bc_connector.get_node_reputation(node_address)
        reputation = float(reputation_bc if reputation_bc is not None else 0)
    features[1] = reputation
    
    # Feature 3 & 4: Transaction count and distinct counterparties
    tx_count, distinct_counterparties = get_node_transaction_count_and_counterparties(bc_connector, node_address)
    features[2] = float(tx_count)
    features[3] = float(distinct_counterparties)

    # Feature 5: Verification Status (Sybils might be set as verified during simulation)
    is_verified_status = bc_connector.is_node_verified(node_address)
    features[4] = 1.0 if is_verified_status else 0.0

    # Labeling: If it's a known Sybil from the log, label it as 1 (Sybil).
    # Otherwise, use heuristic rules or FL model prediction.
    if is_known_sybil:
        label_val = 1
        logger.info(f"Node {node_address} labeled as SYBIL (1) based on sybil_attack_log.json.")
    elif fl_model is not None:
        try:
            # Reshape features for model input if needed
            model_input = features.reshape(1, -1) if hasattr(fl_model, 'predict') else features
            sybil_probability = fl_model.predict(model_input)
            if isinstance(sybil_probability, np.ndarray) and sybil_probability.size > 0:
                sybil_probability = sybil_probability[0]
            label_val = 1 if sybil_probability > 0.5 else 0
            logger.info(f"Node {node_address} - FL Model Prediction: {sybil_probability:.4f}, Label: {'SYBIL' if label_val == 1 else 'NORMAL'}")
        except Exception as e:
            logger.error(f"Error using FL model for {node_address}: {e}. Falling back to heuristic rules.")
            label_val = apply_heuristic_rules(features, is_verified_status, reputation)
    else:
        label_val = apply_heuristic_rules(features, is_verified_status, reputation)
        logger.info(f"Node {node_address} labeled as {'SYBIL' if label_val == 1 else 'NORMAL'} ({label_val}) based on heuristic rules.")
    
    # Check for bribed nodes in Scenario D
    if not is_known_sybil and sybil_attack_log and "scenarioD" in sybil_attack_log and "bribedNodes" in sybil_attack_log["scenarioD"]:
        for bribed_node in sybil_attack_log["scenarioD"]["bribedNodes"]:
            if bribed_node.get("address") == node_address:
                logger.info(f"Node {node_address} identified as a bribed node from scenarioD.")
                # Bribed nodes aren't necessarily Sybils, but we might want to flag them
                # Uncomment the next line if you want to label bribed nodes as Sybils
                # label_val = 1
                break
    
    # Also check flIntegrationMetadata for bribed nodes
    if not is_known_sybil and sybil_attack_log and "flIntegrationMetadata" in sybil_attack_log:
        if "bribedNodeAddresses" in sybil_attack_log["flIntegrationMetadata"]:
            if node_address in sybil_attack_log["flIntegrationMetadata"]["bribedNodeAddresses"]:
                logger.info(f"Node {node_address} identified as a bribed node from flIntegrationMetadata.")
                # Uncomment the next line if you want to label bribed nodes as Sybils
                # label_val = 1
    
    label = np.array([label_val], dtype=np.int32)
    return features, label

def apply_heuristic_rules(features: np.ndarray, is_verified_status: bool, reputation: float) -> int:
    """
    Apply heuristic rules to determine if a node is Sybil
    
    Args:
        features: Feature array for the node
        is_verified_status: Whether the node is verified
        reputation: Node reputation
        
    Returns:
        1 if node is likely a Sybil, 0 otherwise
    """
    sybil_score = 0
    
    # Heuristic rules with weights
    if not is_verified_status:
        sybil_score += 2  # Higher weight for unverified nodes
    
    if features[0] < SYBIL_MAX_AGE_SECONDS and features[0] > 0:
        sybil_score += 1
    elif features[0] == 0 and is_verified_status:
        sybil_score += 0

    if (reputation if reputation is not None else 0) < SYBIL_MAX_REPUTATION:
        sybil_score += 1.5  # Higher weight for low reputation
    
    if features[2] < SYBIL_MAX_TX_COUNT:
        sybil_score += 1
    
    if features[3] < SYBIL_MAX_COUNTERPARTIES:
        sybil_score += 1

    # Adjust threshold for better accuracy
    if is_verified_status and (reputation if reputation is not None else 0) > SYBIL_MAX_REPUTATION:
        return 0  # Verified node with high reputation
    else:
        return 1 if sybil_score >= 4 else 0  # Threshold to reduce false positives

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

    for node_addr in target_node_addresses:
        logger.info(f"Client {client_id}: Processing node {node_addr}")
        try:
            features, label = generate_features_and_label_for_sybil(bc_connector, node_addr, sybil_attack_log=sybil_attack_log)
            client_features_list.append(features)
            client_labels_list.append(label)
            # Small delay to avoid overwhelming the blockchain node
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error processing node {node_addr} for client {client_id}: {e}")

    if not client_features_list:
        logger.warning(f"No valid data for client {client_id}. Returning empty dataset.")
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
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

def make_federated_data_sybil_real(
    all_node_addresses: List[str], 
    num_fl_clients: int = 3, 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> List[tf.data.Dataset]:
    """
    Create federated datasets for Sybil detection using real blockchain data
    
    Args:
        all_node_addresses: List of all node addresses to process
        num_fl_clients: Number of federated learning clients
        sybil_attack_log: Optional attack log data
        
    Returns:
        List of TensorFlow datasets, one for each client
    """
    logger.info(f"Creating federated data for Sybil detection with {num_fl_clients} clients")
    
    # Try to load sybil_attack_log if not provided
    if sybil_attack_log is None:
        sybil_attack_log = find_sybil_attack_log()
    
    # Load environment variables
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "w3storage-upload-script", "ifps_qr.env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    ".env")
    ]
    
    rpc_url = None
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            rpc_url = os.getenv("POLYGON_AMOY_RPC")
            if rpc_url:
                logger.info(f"Loaded RPC URL from {env_path}")
                break
    
    if not rpc_url:
        rpc_url = RPC_URL_OVERRIDE
        logger.warning(f"RPC URL not found in environment variables. Using default: {rpc_url}")

    # Initialize BlockchainConnector
    # Use the RPC_URL_OVERRIDE if provided, otherwise it will use default from .env or hardcoded
    rpc_url = RPC_URL_OVERRIDE if RPC_URL_OVERRIDE else None
    bc_connector = BlockchainConnector(rpc_url=rpc_url)
    
    if not bc_connector.web3 or not bc_connector.contract:
        logger.error("BlockchainConnector not initialized properly. Check RPC URL and network connection.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if "SupplyChainNFT" not in bc_connector.contracts:
        logger.error("SupplyChainNFT contract not loaded. Cannot proceed with real data.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if not all_node_addresses:
        logger.warning("No target node addresses provided. Cannot proceed with real data.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    # Distribute all_node_addresses among clients
    nodes_per_client = len(all_node_addresses) // num_fl_clients
    extra_nodes = len(all_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_data_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(all_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(all_node_addresses))
            client_node_list = all_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list: 
            logger.warning(f"No nodes assigned to client {client_id}. Creating empty dataset.")
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client(client_id, client_node_list, bc_connector, sybil_attack_log=sybil_attack_log)
        client_datasets.append(client_ds)
    
    logger.info(f"Created {len(client_datasets)} federated datasets for Sybil detection")
    return client_datasets

if __name__ == '__main__':
    logger.info("Testing Real Data Preparation for Sybil Detection with Heuristic and Log-based Labeling...")
    
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
            "scenarioA": {},
            "scenarioB": {},
            "scenarioC": {},
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
    federated_datasets = make_federated_data_sybil_real(
        all_node_addresses=test_addresses,
        num_fl_clients=2,
        sybil_attack_log=sybil_log_for_test
    )
    
    logger.info(f"Created {len(federated_datasets)} federated datasets")
    for i, ds in enumerate(federated_datasets):
        for features, labels in ds.take(1):
            logger.info(f"Client {i} dataset sample - Features shape: {features.shape}, Labels shape: {labels.shape}")
