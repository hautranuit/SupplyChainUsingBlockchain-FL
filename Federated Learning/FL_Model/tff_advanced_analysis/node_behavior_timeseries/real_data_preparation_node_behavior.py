#!/usr/bin/env python3
"""
Real Data Preparation for Node Behavior Timeseries FL Model (Phase 3)

This module prepares real blockchain data for the node behavior timeseries federated learning model.
It extracts time-series features from blockchain data to detect anomalous behavior patterns over time.
"""

import tensorflow as tf
import numpy as np
import time
import sys
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_behavior.log'))
    ]
)
logger = logging.getLogger("node_behavior")

# Adjust path to import connectors
current_dir = os.path.dirname(os.path.abspath(__file__))
fl_model_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(fl_model_dir)

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_TS_FEATURES = 4  # tx_frequency, tx_value_sum, new_counterparties, is_flagged_malicious_period
TIMESTEPS = 30  # e.g., 30 days or 30 time periods
PERIOD_DURATION_SECONDS = 86400  # Duration of each timestep in seconds (e.g., 1 day)

ELEMENT_SPEC_P3_TS = (
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32),
    # For an autoencoder, the "label" is the input sequence itself, aiming for reconstruction.
    # If we add a supervised anomaly flag, the label might change or be an additional output.
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32) 
)

# Default RPC URL if not provided in environment
RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

# Cache for timeseries events to avoid repeated blockchain queries
TS_EVENT_CACHE = {
    "PaymentAndTransferCompleted": None,
    "ProductMinted": None,
    "InitialCIDStored": None,
    "DirectSaleAndTransferCompleted": None,
    "DisputeInitiated": None,
    "BatchProposed": None,
    "BatchValidated": None,
    "ArbitratorVoted": None,
    "NodeVerified": None, 
    "all_events_sorted": None, 
    "processed": False
}

RELEVANT_EVENT_NAMES_FOR_TS = [
    "PaymentAndTransferCompleted", "ProductMinted", "InitialCIDStored",
    "DirectSaleAndTransferCompleted", "DisputeInitiated", "BatchProposed",
    "BatchValidated", "ArbitratorVoted", "NodeVerified"
]

def fetch_and_cache_timeseries_events(bc_connector: BlockchainConnector) -> None:
    """
    Fetch and cache timeseries events from the blockchain
    
    Args:
        bc_connector: Blockchain connector instance
    """
    if TS_EVENT_CACHE["processed"]:
        return
    
    logger.info("Fetching and caching events for Node Behavior Timeseries...")
    all_fetched_events = []
    try:
        for event_name in RELEVANT_EVENT_NAMES_FOR_TS:
            events = bc_connector.get_events(contract_name="SupplyChainNFT", event_name=event_name, from_block=0)
            TS_EVENT_CACHE[event_name] = events
            logger.info(f"Fetched {len(events)} {event_name} events.")
            
            for event_data in events:
                try:
                    block_info = bc_connector.w3.eth.get_block(event_data["blockNumber"])
                    # Convert AttributeDict to dict to allow item assignment
                    event_data_dict = dict(event_data)
                    event_data_dict["timestamp"] = block_info["timestamp"]
                    all_fetched_events.append(event_data_dict)
                except Exception as e:
                    logger.warning(f"Could not get timestamp for block {event_data['blockNumber']} of event {event_name}: {e}")
                    event_data["timestamp"] = 0 
        
        all_fetched_events.sort(key=lambda x: x.get("timestamp", 0))
        TS_EVENT_CACHE["all_events_sorted"] = all_fetched_events
        TS_EVENT_CACHE["processed"] = True
        logger.info(f"Total relevant events processed and sorted: {len(all_fetched_events)}")
    except Exception as e:
        logger.error(f"Error fetching timeseries events: {e}")
        TS_EVENT_CACHE["processed"] = False
        for key in TS_EVENT_CACHE:
            if key not in ["all_events_sorted", "processed"]:
                TS_EVENT_CACHE[key] = []
            elif key == "all_events_sorted":
                TS_EVENT_CACHE[key] = None
            else:
                TS_EVENT_CACHE[key] = False

def get_node_start_timestamp(node_address_checksum: str, events_cache: Dict[str, Any]) -> int:
    """
    Get the timestamp when a node was first active on the blockchain
    
    Args:
        node_address_checksum: Checksummed address of the node
        events_cache: Cache of blockchain events
        
    Returns:
        Unix timestamp of first activity, or current time if not found
    """
    node_verified_events = events_cache.get("NodeVerified", [])
    actor_specific_events = [e for e in node_verified_events if e["args"].get("node") == node_address_checksum and e.get("timestamp", 0) > 0]
    
    if actor_specific_events:
        return min(e["timestamp"] for e in actor_specific_events)
    
    first_activity_ts = float("inf")
    found_activity = False
    
    if events_cache.get("all_events_sorted"):
        for event in events_cache["all_events_sorted"]:
            if event.get("timestamp", 0) > 0:
                for arg_name, arg_val in event["args"].items():
                    if isinstance(arg_val, str) and arg_val.lower() == node_address_checksum.lower():
                        first_activity_ts = min(first_activity_ts, event["timestamp"])
                        found_activity = True
                        # Optimization: if we found an event for this node, we are looking for the *absolute earliest*
                        # So we cannot break here, must check all events for this node.
        
        if found_activity and first_activity_ts != float("inf"):
            return first_activity_ts
    
    return int(time.time())

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

def generate_timeseries_for_node(
    node_address_checksum: str, 
    events_cache: Dict[str, Any], 
    w3_instance: Any, 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time-series features for a node
    
    Args:
        node_address_checksum: Checksummed address of the node
        events_cache: Cache of blockchain events
        w3_instance: Web3 instance
        sybil_attack_log: Optional attack log data
        
    Returns:
        Tuple of (input sequence, target sequence) for autoencoder
    """
    sequence = np.zeros((TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
    
    if not events_cache["processed"] or not events_cache.get("all_events_sorted"):
        return sequence, sequence 

    node_start_time = get_node_start_timestamp(node_address_checksum, events_cache)
    current_processing_time = int(time.time())
    
    if sybil_attack_log and sybil_attack_log.get("simulationEndTime"):
        try:
            # Use simulation end time as a more stable reference if available
            current_processing_time = parse_iso_timestamp(sybil_attack_log["simulationEndTime"])
        except Exception as e:
            logger.warning(f"Error parsing simulationEndTime: {e}. Using current time.")
            # Stick to current time if parsing fails

    end_of_last_period = current_processing_time
    known_counterparties_for_node = set()

    # Pre-process sybil log for this node to quickly check if an action falls into a period
    malicious_action_timestamps_by_node = defaultdict(list)
    
    if sybil_attack_log:
        # Check flIntegrationMetadata first for more structured data
        if "flIntegrationMetadata" in sybil_attack_log:
            metadata = sybil_attack_log["flIntegrationMetadata"]
            
            # Check if node is a known Sybil
            if "sybilNodeAddresses" in metadata and node_address_checksum in metadata["sybilNodeAddresses"]:
                logger.info(f"Node {node_address_checksum} is a known Sybil node from flIntegrationMetadata")
                # If we have timestamps for Sybil actions, add them
                if "sybilActionTimestamps" in metadata:
                    for ts_entry in metadata["sybilActionTimestamps"]:
                        if ts_entry.get("nodeAddress") == node_address_checksum and ts_entry.get("timestamp"):
                            try:
                                ts = parse_iso_timestamp(ts_entry["timestamp"])
                                if ts > 0:
                                    malicious_action_timestamps_by_node[node_address_checksum].append(ts)
                            except Exception as e:
                                logger.error(f"Error parsing Sybil action timestamp: {e}")
            
            # Check if node is a known bribed node
            if "bribedNodeAddresses" in metadata and node_address_checksum in metadata["bribedNodeAddresses"]:
                logger.info(f"Node {node_address_checksum} is a known bribed node from flIntegrationMetadata")
                # If we have timestamps for bribe actions, add them
                if "briberyActionTimestamps" in metadata:
                    for ts_entry in metadata["briberyActionTimestamps"]:
                        if ts_entry.get("nodeAddress") == node_address_checksum and ts_entry.get("timestamp"):
                            try:
                                ts = parse_iso_timestamp(ts_entry["timestamp"])
                                if ts > 0:
                                    malicious_action_timestamps_by_node[node_address_checksum].append(ts)
                            except Exception as e:
                                logger.error(f"Error parsing bribery action timestamp: {e}")
        
        # Process detailed scenario data
        for scenario_key in ['scenarioA', 'scenarioB', 'scenarioC', 'scenarioD']:
            scenario = sybil_attack_log.get(scenario_key, {})
            
            if scenario_key == 'scenarioD': # Bribery actions are logged differently
                # Consider bribe payments themselves or simulated behaviors
                for action in scenario.get('actions', []):
                    if action.get('status') == 'success' and w3_instance.to_checksum_address(action.get('targetAddress', '')) == node_address_checksum:
                        try:
                            ts = parse_iso_timestamp(action['timestamp'])
                            if ts > 0:
                                malicious_action_timestamps_by_node[node_address_checksum].append(ts)
                        except Exception as e:
                            logger.error(f"Error parsing bribery action timestamp: {e}")
                
                for change in scenario.get('simulatedBehavioralChanges', []):
                    if w3_instance.to_checksum_address(change.get('actor', '')) == node_address_checksum:
                        try:
                            ts = parse_iso_timestamp(change['timestamp'])
                            if ts > 0:
                                malicious_action_timestamps_by_node[node_address_checksum].append(ts)
                        except Exception as e:
                            logger.error(f"Error parsing behavioral change timestamp: {e}")
            else: # Scenarios A, B, C
                for action in scenario.get('actions', []):
                    actor = action.get('sybilNode') or action.get('proposerNode') or action.get('validatorNode')
                    if actor and w3_instance.to_checksum_address(actor) == node_address_checksum and action.get('status') == 'success':
                        try:
                            ts = parse_iso_timestamp(action['timestamp'])
                            if ts > 0:
                                malicious_action_timestamps_by_node[node_address_checksum].append(ts)
                        except Exception as e:
                            logger.error(f"Error parsing scenario action timestamp: {e}")

    for i in range(TIMESTEPS):
        period_end_time = end_of_last_period - (i * PERIOD_DURATION_SECONDS)
        period_start_time = period_end_time - PERIOD_DURATION_SECONDS
        
        if period_start_time < node_start_time - PERIOD_DURATION_SECONDS: 
            continue

        tx_frequency = 0
        tx_value_sum = 0.0
        current_period_counterparties = set()
        is_flagged_malicious_this_period = 0.0

        # Check if any logged malicious action for this node falls into the current period
        if node_address_checksum in malicious_action_timestamps_by_node:
            for mal_ts in malicious_action_timestamps_by_node[node_address_checksum]:
                if period_start_time <= mal_ts < period_end_time:
                    is_flagged_malicious_this_period = 1.0
                    logger.info(f"Node {node_address_checksum} has malicious action in period {i+1} (timestamp {mal_ts})")
                    break

        for event in events_cache["all_events_sorted"]:
            event_ts = event.get("timestamp", 0)
            if not (period_start_time <= event_ts < period_end_time):
                continue

            is_node_involved = False
            event_value = 0.0
            counterparty_in_event = None
            args = event.get("args", {})

            # Process different event types
            if event["event"] == "PaymentAndTransferCompleted":
                if args.get("seller") == node_address_checksum:
                    is_node_involved = True
                    event_value = float(args.get("price", 0))
                    counterparty_in_event = args.get("buyer")
                elif args.get("buyer") == node_address_checksum:
                    is_node_involved = True
                    event_value = float(args.get("price", 0))
                    counterparty_in_event = args.get("seller")
            
            elif event["event"] == "ProductMinted" and args.get("owner") == node_address_checksum:
                is_node_involved = True
            
            elif event["event"] == "InitialCIDStored" and args.get("actor") == node_address_checksum:
                is_node_involved = True
            
            elif event["event"] == "DirectSaleAndTransferCompleted":
                if args.get("seller") == node_address_checksum:
                    is_node_involved = True
                    counterparty_in_event = args.get("buyer")
                elif args.get("buyer") == node_address_checksum:
                    is_node_involved = True
                    counterparty_in_event = args.get("seller")
            
            elif event["event"] == "DisputeInitiated":
                if args.get("initiator") == node_address_checksum:
                    is_node_involved = True
                    counterparty_in_event = args.get("currentOwner")
                elif args.get("currentOwner") == node_address_checksum: 
                    is_node_involved = True
                    counterparty_in_event = args.get("initiator")
            
            elif event["event"] == "BatchProposed" and args.get("proposer") == node_address_checksum:
                is_node_involved = True 
            
            elif event["event"] == "BatchValidated" and args.get("validator") == node_address_checksum:
                is_node_involved = True
            
            elif event["event"] == "ArbitratorVoted" and args.get("voter") == node_address_checksum:
                is_node_involved = True
            
            elif event["event"] == "NodeVerified" and args.get("node") == node_address_checksum:
                is_node_involved = True

            if is_node_involved:
                tx_frequency += 1
                tx_value_sum += event_value
                
                if counterparty_in_event and isinstance(counterparty_in_event, str):
                    try: # Ensure counterparty address is valid before checksumming
                        checksummed_counterparty = w3_instance.to_checksum_address(counterparty_in_event)
                        if checksummed_counterparty != node_address_checksum:
                            current_period_counterparties.add(checksummed_counterparty)
                    except ValueError: # Invalid address format
                        logger.warning(f"Invalid counterparty address format encountered: {counterparty_in_event}")
        
        new_counterparties_this_period = len(current_period_counterparties - known_counterparties_for_node)
        known_counterparties_for_node.update(current_period_counterparties)
        
        sequence_idx = TIMESTEPS - 1 - i # Store features chronologically (oldest to newest)
        sequence[sequence_idx, 0] = float(tx_frequency)
        sequence[sequence_idx, 1] = float(tx_value_sum) 
        sequence[sequence_idx, 2] = float(new_counterparties_this_period)
        sequence[sequence_idx, 3] = float(is_flagged_malicious_this_period) # Malicious activity flag

    return sequence, sequence 

def load_real_data_for_fl_client_timeseries(
    client_id: str, 
    client_info: Dict[str, Any], 
    bc_connector: BlockchainConnector, 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> tf.data.Dataset:
    """
    Load real data for a federated learning client
    
    Args:
        client_id: Identifier for the client
        client_info: Dictionary with client information (address, role, etc.)
        bc_connector: Blockchain connector instance
        sybil_attack_log: Optional attack log data
        
    Returns:
        TensorFlow dataset for the client
    """
    node_addr_orig = client_info.get("address")
    logger.info(f"FL Client {client_id} ({node_addr_orig}): Loading real timeseries data")
    
    if not TS_EVENT_CACHE["processed"]:
        fetch_and_cache_timeseries_events(bc_connector)

    client_input_seqs = []
    client_target_seqs = [] 

    try:
        node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr_orig)
        input_s, target_s = generate_timeseries_for_node(node_addr_checksum, TS_EVENT_CACHE, bc_connector.w3, sybil_attack_log)
        client_input_seqs.append(input_s)
        client_target_seqs.append(target_s)
        logger.info(f"Generated timeseries for node {node_addr_checksum}")
    except Exception as e:
        logger.error(f"Error processing node {node_addr_orig} for client {client_id} timeseries: {e}")
        import traceback
        logger.error(traceback.format_exc())

    if not client_input_seqs:
        logger.warning(f"No valid data for client {client_id}. Returning empty dataset.")
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        )).batch(1)

    input_array = np.array(client_input_seqs, dtype=np.float32)
    target_array = np.array(client_target_seqs, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((input_array, target_array))
    # Each client gets data for one node, so batch size 1 is appropriate here.
    logger.info(f"Client {client_id}: Created dataset with {len(client_input_seqs)} samples")
    return dataset.batch(1) 

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

def make_federated_data_p3_timeseries_real(
    clients_info: List[Dict[str, Any]], 
    sybil_attack_log: Optional[Dict[str, Any]] = None
) -> List[tf.data.Dataset]:
    """
    Create federated datasets for node behavior timeseries using real blockchain data
    
    Args:
        clients_info: List of dictionaries with client information
        sybil_attack_log: Optional attack log data
        
    Returns:
        List of TensorFlow datasets, one for each client
    """
    logger.info(f"Creating federated data for node behavior timeseries with {len(clients_info)} clients")
    
    # Try to load sybil_attack_log if not provided
    if sybil_attack_log is None:
        sybil_attack_log = find_sybil_attack_log()
    
    # Initialize BlockchainConnector
    # Use the RPC_URL_OVERRIDE if provided, otherwise it will use default from .env or hardcoded
    rpc_url = RPC_URL_OVERRIDE if RPC_URL_OVERRIDE else None
    bc_connector = BlockchainConnector(rpc_url=rpc_url)

    if not bc_connector.web3 or not bc_connector.contract:
        logger.error("BlockchainConnector not properly initialized. Exiting.")
        return []

    # Ensure timeseries events are cached
    fetch_and_cache_timeseries_events(bc_connector)

    # Create dataset for each client
    client_datasets = []
    for i, client_info in enumerate(clients_info):
        client_id = f"timeseries_client_{i}"
        client_ds = load_real_data_for_fl_client_timeseries(client_id, client_info, bc_connector, sybil_attack_log)
        client_datasets.append(client_ds)
    
    logger.info(f"Created {len(client_datasets)} federated datasets for node behavior timeseries")
    return client_datasets

if __name__ == '__main__':
    logger.info("Testing Real Data Preparation for Node Behavior Timeseries with Attack Log Integration...")
    
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
                        "type": "disputeCreation",
                        "tokenId": "1",
                        "sybilNode": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
                        "timestamp": "2024-01-02T10:00:00.000Z",
                        "status": "success"
                    }
                ]
            },
            "flIntegrationMetadata": {
                "sybilNodeAddresses": ["0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"],
                "sybilActionTimestamps": [
                    {"nodeAddress": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", "timestamp": "2024-01-02T10:00:00.000Z"}
                ]
            }
        }
    
    # Test with a few sample addresses
    test_clients = [
        {"address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8", "role": "producer"},
        {"address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC", "role": "transporter"},
        {"address": "0x90F79bf6EB2c4f870365E785982E1f101E93b906", "role": "retailer"}
    ]
    
    if sybil_log_for_test and "sybilNodes" in sybil_log_for_test:
        for node in sybil_log_for_test["sybilNodes"]:
            if "address" in node and not any(client["address"] == node["address"] for client in test_clients):
                test_clients.append({"address": node["address"], "role": "unknown"})
    
    logger.info(f"Testing with clients: {test_clients}")
    
    # Create federated datasets
    federated_datasets = make_federated_data_p3_timeseries_real(
        clients_info=test_clients,
        sybil_attack_log=sybil_log_for_test
    )
    
    logger.info(f"Created {len(federated_datasets)} federated datasets")
    for i, ds in enumerate(federated_datasets):
        for features, labels in ds.take(1):
            logger.info(f"Client {i} dataset sample - Features shape: {features.shape}, Labels shape: {labels.shape}")
