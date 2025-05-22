#!/usr/bin/env python3
"""
Real Data Preparation for Dispute Risk FL Model (Phase 3)

This module prepares real blockchain data for the dispute risk federated learning model.
It extracts features from blockchain data to predict the risk level of disputes.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dispute_risk.log'))
    ]
)
logger = logging.getLogger("dispute_risk")

# Adjust path to import connectors
current_dir = os.path.dirname(os.path.abspath(__file__))
fl_model_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(fl_model_dir)

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_DR_FEATURES = 6 
ELEMENT_SPEC_P3_DR = (
    tf.TensorSpec(shape=(NUM_P3_DR_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for low risk, 1 for high risk
)

# Default RPC URL if not provided in environment
RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

# Cache for dispute events to avoid repeated blockchain queries
DISPUTE_EVENT_CACHE_DR = {
    "DisputeInitiated": None,
    "ProductListedForSale": None, 
    "CollateralDepositedForPurchase": None, 
    "CIDStored": None, 
    "NodeVerified": None, 
    "processed": False
}

# Heuristic thresholds for Dispute Risk labeling (can be tuned)
HIGH_DISPUTE_VALUE_THRESHOLD = 10000  # Example: Value in wei or smallest unit
LARGE_REPUTATION_DIFFERENCE_THRESHOLD = 50
LOW_EVIDENCE_COUNT_THRESHOLD = 2 # Fewer than 2 pieces of evidence
INITIATOR_HIGH_PRIOR_RISK_THRESHOLD = 0.8 # Based on normalized inverse age (closer to 1 is higher risk)
LOW_AVERAGE_REPUTATION_THRESHOLD = 20
MIN_INITIATOR_AGE_FOR_LOW_RISK_SECONDS = 86400 * 14 # e.g. initiator must be older than 2 weeks to be considered lower risk by default
SYBIL_DISPUTE_RISK_INCREASE = 3 # Additional risk score for disputes initiated by known Sybils/malicious actors

def fetch_and_cache_dispute_risk_events(bc_connector: BlockchainConnector) -> None:
    """
    Fetch and cache dispute risk events from the blockchain
    
    Args:
        bc_connector: Blockchain connector instance
    """
    if DISPUTE_EVENT_CACHE_DR["processed"]:
        return
    
    logger.info("Fetching and caching events for dispute risk analysis...")
    try:
        DISPUTE_EVENT_CACHE_DR["DisputeInitiated"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="DisputeInitiated", from_block=0
        )
        DISPUTE_EVENT_CACHE_DR["ProductListedForSale"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="ProductListedForSale", from_block=0
        )
        DISPUTE_EVENT_CACHE_DR["CollateralDepositedForPurchase"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="CollateralDepositedForPurchase", from_block=0
        )
        DISPUTE_EVENT_CACHE_DR["CIDStored"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="CIDStored", from_block=0
        )
        DISPUTE_EVENT_CACHE_DR["NodeVerified"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="NodeVerified", from_block=0
        )
        DISPUTE_EVENT_CACHE_DR["processed"] = True
        logger.info(f"Fetched {len(DISPUTE_EVENT_CACHE_DR['DisputeInitiated'])} DisputeInitiated events for DR.")
    except Exception as e:
        logger.error(f"Error fetching dispute risk events: {e}")
        DISPUTE_EVENT_CACHE_DR["processed"] = False
        for key in DISPUTE_EVENT_CACHE_DR:
            DISPUTE_EVENT_CACHE_DR[key] = [] if key != "processed" else False

def get_dispute_value(token_id: str, events_cache: Dict[str, Any]) -> float:
    """
    Get the value of a dispute based on collateral or listing price
    
    Args:
        token_id: Token ID of the disputed item
        events_cache: Cache of blockchain events
        
    Returns:
        Value of the dispute
    """
    collateral_event = next((e for e in reversed(events_cache.get("CollateralDepositedForPurchase", [])) if e["args"]["tokenId"] == token_id), None)
    if collateral_event:
        return float(collateral_event["args"]["amount"])
    
    listing_event = next((e for e in reversed(events_cache.get("ProductListedForSale", [])) if e["args"]["tokenId"] == token_id), None)
    if listing_event:
        return float(listing_event["args"]["price"])
    
    return 0.0

def get_node_age_seconds(node_address_checksum: str, events_cache: Dict[str, Any], current_timestamp: int, w3_instance: Any) -> float:
    """
    Get the age of a node in seconds
    
    Args:
        node_address_checksum: Checksummed address of the node
        events_cache: Cache of blockchain events
        current_timestamp: Current Unix timestamp
        w3_instance: Web3 instance
        
    Returns:
        Age of the node in seconds
    """
    verified_events = [e for e in events_cache.get("NodeVerified", []) if e["args"]["node"] == node_address_checksum]
    if not verified_events:
        return 0
    
    verified_events.sort(key=lambda x: x["blockNumber"])
    first_verification_event = verified_events[0]
    block_number = first_verification_event["blockNumber"]
    
    try:
        timestamp = w3_instance.eth.get_block(block_number)["timestamp"]
        return float(current_timestamp - timestamp)
    except Exception as e:
        logger.error(f"Error getting node age for {node_address_checksum}: {e}")
        return 0

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

def generate_features_and_label_for_dispute(
    dispute_event: Dict[str, Any], 
    bc_connector: BlockchainConnector, 
    events_cache: Dict[str, Any], 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate features and label for dispute risk for a single dispute
    
    Args:
        dispute_event: Dispute event data
        bc_connector: Blockchain connector instance
        events_cache: Cache of blockchain events
        sybil_attack_log: Optional attack log data
        
    Returns:
        Tuple of (features array, label array)
    """
    features = np.zeros(NUM_P3_DR_FEATURES, dtype=np.float32)
    risk_score = 0
    current_time = int(time.time())

    token_id = dispute_event["args"]["tokenId"]
    initiator_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["initiator"])
    defendant_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["currentOwner"])

    # F0: Dispute Value
    dispute_value = get_dispute_value(token_id, events_cache)
    features[0] = dispute_value
    if dispute_value > HIGH_DISPUTE_VALUE_THRESHOLD:
        risk_score += 2
        logger.info(f"Dispute for token {token_id} has high value: {dispute_value}")

    # F1: Number of Parties (currently fixed at 2)
    features[1] = 2.0 

    # F2: Average Reputation of Parties
    rep_initiator = bc_connector.get_node_reputation(initiator_checksum) or 0
    rep_defendant = bc_connector.get_node_reputation(defendant_checksum) or 0
    avg_reputation = (float(rep_initiator) + float(rep_defendant)) / 2.0
    features[2] = avg_reputation
    if avg_reputation < LOW_AVERAGE_REPUTATION_THRESHOLD:
        risk_score += 1
        logger.info(f"Dispute for token {token_id} has low average reputation: {avg_reputation}")

    # F3: Reputation Difference
    reputation_diff = abs(float(rep_initiator) - float(rep_defendant))
    features[3] = reputation_diff
    if reputation_diff > LARGE_REPUTATION_DIFFERENCE_THRESHOLD:
        risk_score += 1
        logger.info(f"Dispute for token {token_id} has large reputation difference: {reputation_diff}")

    # F4: Evidence Count
    evidence_count = sum(1 for e in events_cache.get("CIDStored", []) if e["args"]["tokenId"] == token_id)
    features[4] = float(evidence_count)
    if evidence_count < LOW_EVIDENCE_COUNT_THRESHOLD:
        risk_score += 1
        logger.info(f"Dispute for token {token_id} has low evidence count: {evidence_count}")

    # F5: Initiator Prior Risk (based on age)
    initiator_age_seconds = get_node_age_seconds(initiator_checksum, events_cache, current_time, bc_connector.w3)
    initiator_prior_risk_feature_val = 1.0 # Max risk if age is 0 or very new
    if initiator_age_seconds > 0:
        # Normalize age to months, then inverse+1. Max value is 1 (for age=0), decreases as age increases.
        initiator_prior_risk_feature_val = 1.0 / ((initiator_age_seconds / (86400 * 30)) + 1.0)
    features[5] = initiator_prior_risk_feature_val
    
    if initiator_prior_risk_feature_val > INITIATOR_HIGH_PRIOR_RISK_THRESHOLD: # Higher value means newer node
        risk_score += 2
        logger.info(f"Dispute for token {token_id} has high initiator prior risk: {initiator_prior_risk_feature_val}")
    elif initiator_age_seconds == 0: # If age is truly zero (couldn't be found), also high risk
        risk_score += 2
        logger.info(f"Dispute for token {token_id} has initiator with unknown age")
    
    # Additional check: if initiator is very new (e.g. < 2 weeks), it might be a risky dispute
    if 0 < initiator_age_seconds < MIN_INITIATOR_AGE_FOR_LOW_RISK_SECONDS:
        risk_score += 1
        logger.info(f"Dispute for token {token_id} has very new initiator: {initiator_age_seconds / 86400:.1f} days old")

    # Incorporate Sybil attack log information
    is_malicious_dispute = False
    
    # Check if initiator is a known Sybil or bribed node
    if is_known_sybil_or_bribed_node(initiator_checksum, sybil_attack_log):
        risk_score += SYBIL_DISPUTE_RISK_INCREASE
        is_malicious_dispute = True
        logger.info(f"Dispute for token {token_id} initiated by known Sybil/bribed node {initiator_checksum}")
    
    if sybil_attack_log:
        # Check Scenario A: Frivolous disputes by Sybils
        if 'scenarioA' in sybil_attack_log and 'actions' in sybil_attack_log['scenarioA']:
            for action in sybil_attack_log['scenarioA']['actions']:
                if action.get('type') == 'initiateFrivolousDispute' and \
                   action.get('status') == 'success' and \
                   bc_connector.w3.to_checksum_address(action.get('sybilNode', '')) == initiator_checksum and \
                   action.get('tokenId') == token_id:
                    risk_score += SYBIL_DISPUTE_RISK_INCREASE
                    is_malicious_dispute = True
                    logger.info(f"Dispute for token {token_id} by {initiator_checksum} (Scenario A) identified. Risk increased.")
                    break
        
        # Check Scenario D: Disputes potentially initiated by bribed nodes
        if not is_malicious_dispute and 'scenarioD' in sybil_attack_log and 'bribedNodes' in sybil_attack_log['scenarioD']:
            for bribed_node_info in sybil_attack_log['scenarioD']['bribedNodes']:
                if bc_connector.w3.to_checksum_address(bribed_node_info.get('address', '')) == initiator_checksum:
                    # Check for simulated behavioral changes related to disputes
                    for change in sybil_attack_log['scenarioD'].get('simulatedBehavioralChanges', []):
                        if bc_connector.w3.to_checksum_address(change.get('actor', '')) == initiator_checksum:
                            if any("dispute" in detail.lower() for detail in change.get('details', [])):
                                risk_score += SYBIL_DISPUTE_RISK_INCREASE
                                is_malicious_dispute = True
                                logger.info(f"Dispute for token {token_id} by bribed node {initiator_checksum} (Scenario D behavior) identified. Risk increased.")
                                break
                    if is_malicious_dispute:
                        break

    label_val = 1 if risk_score >= 3 or is_malicious_dispute else 0 # Threshold for labeling as high-risk dispute
    label = np.array([label_val], dtype=np.int32)

    if label_val == 1:
        logger.info(f"Dispute for token {token_id} labeled HIGH RISK (score: {risk_score}). " +
                   f"Val:{features[0]:.0f}, AvgRep:{features[2]:.1f}, RepDiff:{features[3]:.1f}, " +
                   f"Evid:{features[4]:.0f}, InitRisk:{features[5]:.2f}, Malicious: {is_malicious_dispute}")
    else:
        logger.info(f"Dispute for token {token_id} labeled LOW RISK (score: {risk_score})")
    
    return features, label

def load_real_data_for_fl_client_dispute_risk(
    client_id: str, 
    client_address: str, 
    all_dispute_events: List[Dict[str, Any]], 
    bc_connector: BlockchainConnector, 
    events_cache: Dict[str, Any], 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> tf.data.Dataset:
    """
    Load real data for a federated learning client
    
    Args:
        client_id: Identifier for the client
        client_address: Address of the client
        all_dispute_events: List of all dispute events
        bc_connector: Blockchain connector instance
        events_cache: Cache of blockchain events
        sybil_attack_log: Optional attack log data
        
    Returns:
        TensorFlow dataset for the client
    """
    logger.info(f"FL Client {client_id} ({client_address}): Loading real dispute risk data")
    client_features_list = []
    client_labels_list = []

    # Filter disputes relevant to this client (initiator or defendant)
    client_specific_disputes = []
    for dispute_event in all_dispute_events:
        initiator_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["initiator"])
        defendant_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["currentOwner"])
        if client_address == initiator_checksum or client_address == defendant_checksum:
            client_specific_disputes.append(dispute_event)
    
    logger.info(f"Client {client_id} has {len(client_specific_disputes)} relevant disputes out of {len(all_dispute_events)} total.")

    for dispute_event in client_specific_disputes:
        try:
            features, label = generate_features_and_label_for_dispute(dispute_event, bc_connector, events_cache, sybil_attack_log)
            client_features_list.append(features)
            client_labels_list.append(label)
            time.sleep(0.01) 
        except Exception as e:
            logger.error(f"Error processing dispute event for client {client_id} (Dispute Risk): {e}")
            import traceback
            logger.error(traceback.format_exc())

    if not client_features_list:
        logger.warning(f"No valid data for client {client_id}. Returning empty dataset.")
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    logger.info(f"Client {client_id}: Created dataset with {len(client_features_list)} samples")
    return dataset.shuffle(len(client_features_list) if len(client_features_list) > 0 else 1).batch(32)

def find_sybil_attack_log() -> Optional[Dict[str, Any]]:
    """
    Find and load the sybil_attack_log.json file
    
    Returns:
        Dictionary containing sybil attack log data, or None if not found
    """
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
                    "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "sybil_attack_log.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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

def make_federated_data_p3_dispute_real(
    all_node_addresses: List[str], 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> List[tf.data.Dataset]:
    """
    Create federated datasets for dispute risk using real blockchain data
    
    Args:
        all_node_addresses: List of all node addresses to process
        sybil_attack_log: Optional attack log data
        
    Returns:
        List of TensorFlow datasets, one for each client
    """
    logger.info(f"Creating federated data for dispute risk with {len(all_node_addresses)} clients")
    
    # Try to load sybil_attack_log if not provided
    if sybil_attack_log is None:
        sybil_attack_log = find_sybil_attack_log()
    
    # Initialize BlockchainConnector
    # Use the RPC_URL_OVERRIDE if provided, otherwise it will use default from .env or hardcoded
    rpc_url = RPC_URL_OVERRIDE if RPC_URL_OVERRIDE else None
    bc_connector = BlockchainConnector(rpc_url=rpc_url)

    if not bc_connector.web3 or not bc_connector.contract:
        logger.error("BlockchainConnector not properly initialized. Exiting.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(len(all_node_addresses))]

    fetch_and_cache_dispute_risk_events(bc_connector)
    if not DISPUTE_EVENT_CACHE_DR["processed"] or not DISPUTE_EVENT_CACHE_DR.get("DisputeInitiated"):
        logger.error("Critical Error: Failed to fetch DisputeInitiated events (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(len(all_node_addresses))]

    all_dispute_events = DISPUTE_EVENT_CACHE_DR["DisputeInitiated"]
    if not all_dispute_events:
        logger.warning("Warning: No DisputeInitiated events found in cache (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(len(all_node_addresses))]
    
    client_datasets = []
    num_fl_clients = len(all_node_addresses)

    # Ensure all node addresses are checksummed for consistent comparison
    checksummed_node_addresses = [bc_connector.w3.to_checksum_address(addr) for addr in all_node_addresses]

    for i, client_node_address in enumerate(checksummed_node_addresses):
        client_id = f"real_dr_client_{i}_{client_node_address[:8]}" # Unique client ID including part of address
        
        # Pass all dispute events; filtering happens in load_real_data_for_fl_client_dispute_risk
        client_ds = load_real_data_for_fl_client_dispute_risk(
            client_id, 
            client_node_address, 
            all_dispute_events, 
            bc_connector, 
            DISPUTE_EVENT_CACHE_DR,
            sybil_attack_log
        )
        client_datasets.append(client_ds)
    
    logger.info(f"Created {len(client_datasets)} client datasets for dispute risk based on {num_fl_clients} node addresses.")
    return client_datasets

# Add alias for backward compatibility
make_federated_data_dispute_risk_real = make_federated_data_p3_dispute_real

if __name__ == '__main__':
    logger.info("Testing Real Data Preparation for Dispute Risk with Heuristic Labeling and Sybil Data...")
    
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
            "scenarioA": {
                "actions": [
                    {
                        "type": "initiateFrivolousDispute",
                        "tokenId": "1",
                        "sybilNode": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        "timestamp": "2024-01-02T10:00:00.000Z",
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
    federated_datasets = make_federated_data_p3_dispute_real(
        all_node_addresses=test_addresses,
        sybil_attack_log=sybil_log_for_test
    )
    
    logger.info(f"Created {len(federated_datasets)} federated datasets")
    for i, ds in enumerate(federated_datasets):
        for features, labels in ds.take(1):
            logger.info(f"Client {i} dataset sample - Features shape: {features.shape}, Labels shape: {labels.shape}")
