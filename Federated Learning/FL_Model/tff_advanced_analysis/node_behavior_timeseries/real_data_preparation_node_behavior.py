# Real Data Preparation for Node Behavior Timeseries FL Model (Phase 3)

import tensorflow as tf
import numpy as np
import time
import sys
import json # Added for sybil_attack_log
from collections import defaultdict, deque

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/")

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_TS_FEATURES = 4  # tx_frequency, tx_value_sum, new_counterparties, is_flagged_malicious_period
TIMESTEPS = 30  # e.g., 30 days or 30 time periods
PERIOD_DURATION_SECONDS = 86400  # Duration of each timestep in seconds (e.g., 1 day)

ELEMENT_SPEC_P3_TS = (
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32),
    # For an autoencoder, the "label" is the input sequence itself, aiming for reconstruction.
    # If we add a supervised anomaly flag, the label might change or be an additional output.
    # For now, keeping it as reconstruction target.
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32) 
)

RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

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

# --- Labeling Strategy for Node Behavior Timeseries (Autoencoder Approach) ---
# For an autoencoder model aimed at anomaly detection, the training process itself doesn't use explicit 0/1 labels (normal/anomalous).
# Instead, the model is trained to reconstruct its input. The input sequences are historical behavior data.
# 1. Feature Engineering: Extract meaningful time-series features (e.g., transaction frequency, value, new counterparties per period).
# 2. Training: The autoencoder learns to compress and then reconstruct these feature sequences. The input sequence is also the target output sequence.
# 3. Anomaly Detection (Post-Training): 
#    - When a new time-series sequence for a node is fed into the trained autoencoder, a reconstruction error is calculated (e.g., Mean Squared Error between input and reconstructed output).
#    - A threshold is set for this reconstruction error. If a sequence's error exceeds this threshold, it is flagged as anomalous.
#    - The threshold can be determined based on the distribution of reconstruction errors on a validation set of known normal data.
# This script focuses on step 1 and preparing data for step 2. The "labeling" is implicit in that the model learns what is "normal" by trying to reconstruct all provided historical data.
# Explicit 0/1 labels for anomalous sequences would be used if training a supervised anomaly detector, or for evaluating the autoencoder's performance on a test set with known anomalies.

def fetch_and_cache_timeseries_events(bc_connector):
    if TS_EVENT_CACHE["processed"]:
        return
    print("Fetching and caching events for Node Behavior Timeseries...")
    all_fetched_events = []
    try:
        for event_name in RELEVANT_EVENT_NAMES_FOR_TS:
            events = bc_connector.get_events(contract_name="SupplyChainNFT", event_name=event_name, from_block=0)
            TS_EVENT_CACHE[event_name] = events
            # print(f"Fetched {len(events)} {event_name} events.")
            for event_data in events:
                try:
                    block_info = bc_connector.w3.eth.get_block(event_data["blockNumber"])
                    event_data["timestamp"] = block_info["timestamp"]
                    all_fetched_events.append(event_data)
                except Exception as e:
                    # print(f"Could not get timestamp for block {event_data["blockNumber"]} of event {event_name}: {e}")
                    event_data["timestamp"] = 0 
        all_fetched_events.sort(key=lambda x: x.get("timestamp", 0))
        TS_EVENT_CACHE["all_events_sorted"] = all_fetched_events
        TS_EVENT_CACHE["processed"] = True
        # print(f"Total relevant events processed and sorted: {len(all_fetched_events)}")
    except Exception as e:
        print(f"Error fetching timeseries events: {e}")
        TS_EVENT_CACHE["processed"] = False
        for key in TS_EVENT_CACHE: TS_EVENT_CACHE[key] = [] if key not in ["all_events_sorted", "processed"] else (None if key == "all_events_sorted" else False)

def get_node_start_timestamp(node_address_checksum, events_cache):
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

def generate_timeseries_for_node(node_address_checksum, events_cache, w3_instance, sybil_attack_log=None):
    sequence = np.zeros((TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
    if not events_cache["processed"] or not events_cache.get("all_events_sorted"):
        return sequence, sequence 

    node_start_time = get_node_start_timestamp(node_address_checksum, events_cache)
    current_processing_time = int(time.time())
    if sybil_attack_log and sybil_attack_log.get("simulationEndTime"):
        try:
            # Use simulation end time as a more stable reference if available
            current_processing_time = int(datetime.fromisoformat(sybil_attack_log["simulationEndTime"].replace('Z', '+00:00')).timestamp())
        except:
            pass # Stick to current time if parsing fails

    end_of_last_period = current_processing_time
    known_counterparties_for_node = set()

    # Pre-process sybil log for this node to quickly check if an action falls into a period
    malicious_action_timestamps_by_node = defaultdict(list)
    if sybil_attack_log:
        for scenario_key in ['scenarioA', 'scenarioB', 'scenarioC', 'scenarioD']:
            scenario = sybil_attack_log.get(scenario_key, {})
            actions = scenario.get('actions', [])
            if scenario_key == 'scenarioD': # Bribery actions are logged differently
                # Consider bribe payments themselves or simulated behaviors
                for action in actions: # Bribe payment attempts
                    if action.get('status') == 'success' and w3_instance.to_checksum_address(action.get('targetAddress', '')) == node_address_checksum:
                        try: malicious_action_timestamps_by_node[node_address_checksum].append(int(datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00')).timestamp()))
                        except: pass
                for change in scenario.get('simulatedBehavioralChanges', []):
                     if w3_instance.to_checksum_address(change.get('actor', '')) == node_address_checksum:
                        try: malicious_action_timestamps_by_node[node_address_checksum].append(int(datetime.fromisoformat(change['timestamp'].replace('Z', '+00:00')).timestamp()))
                        except: pass       
            else: # Scenarios A, B, C
                for action in actions:
                    actor = action.get('sybilNode') or action.get('proposerNode') or action.get('validatorNode') # etc.
                    if actor and w3_instance.to_checksum_address(actor) == node_address_checksum and action.get('status') == 'success':
                        try: malicious_action_timestamps_by_node[node_address_checksum].append(int(datetime.fromisoformat(action['timestamp'].replace('Z', '+00:00')).timestamp()))
                        except: pass

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
                    break

        for event in events_cache["all_events_sorted"]:
            event_ts = event.get("timestamp", 0)
            if not (period_start_time <= event_ts < period_end_time):
                continue

            is_node_involved = False
            event_value = 0.0
            counterparty_in_event = None
            args = event.get("args", {})

            # --- Existing event processing logic to determine is_node_involved, event_value, counterparty_in_event ---
            if event["event"] == "PaymentAndTransferCompleted":
                if args.get("seller") == node_address_checksum:
                    is_node_involved = True; event_value = float(args.get("price", 0)); counterparty_in_event = args.get("buyer")
                elif args.get("buyer") == node_address_checksum:
                    is_node_involved = True; event_value = float(args.get("price", 0)); counterparty_in_event = args.get("seller")
            elif event["event"] == "ProductMinted" and args.get("owner") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "InitialCIDStored" and args.get("actor") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "DirectSaleAndTransferCompleted":
                if args.get("seller") == node_address_checksum:
                    is_node_involved = True; counterparty_in_event = args.get("buyer")
                elif args.get("buyer") == node_address_checksum:
                    is_node_involved = True; counterparty_in_event = args.get("seller")
            elif event["event"] == "DisputeInitiated":
                if args.get("initiator") == node_address_checksum:
                    is_node_involved = True; counterparty_in_event = args.get("currentOwner")
                elif args.get("currentOwner") == node_address_checksum: 
                    is_node_involved = True; counterparty_in_event = args.get("initiator")
            elif event["event"] == "BatchProposed" and args.get("proposer") == node_address_checksum:
                is_node_involved = True 
            elif event["event"] == "BatchValidated" and args.get("validator") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "ArbitratorVoted" and args.get("voter") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "NodeVerified" and args.get("node") == node_address_checksum:
                 is_node_involved = True
            # --- End of existing event processing logic ---

            if is_node_involved:
                tx_frequency += 1
                tx_value_sum += event_value
                if counterparty_in_event and isinstance(counterparty_in_event, str):
                    try: # Ensure counterparty address is valid before checksumming
                        checksummed_counterparty = w3_instance.to_checksum_address(counterparty_in_event)
                        if checksummed_counterparty != node_address_checksum:
                            current_period_counterparties.add(checksummed_counterparty)
                    except ValueError: # Invalid address format
                        # print(f"Warning: Invalid counterparty address format encountered: {counterparty_in_event}")
                        pass # Or log this occurrence
        
        new_counterparties_this_period = len(current_period_counterparties - known_counterparties_for_node)
        known_counterparties_for_node.update(current_period_counterparties)
        
        sequence_idx = TIMESTEPS - 1 - i # Store features chronologically (oldest to newest)
        sequence[sequence_idx, 0] = float(tx_frequency)
        sequence[sequence_idx, 1] = float(tx_value_sum) 
        sequence[sequence_idx, 2] = float(new_counterparties_this_period)
        sequence[sequence_idx, 3] = float(is_flagged_malicious_this_period) # New feature

    return sequence, sequence 

def load_real_data_for_fl_client_timeseries(client_id: str, client_info: dict, bc_connector, sybil_attack_log=None):
    # client_info is expected to be a dict like {'address': '0x...', 'role': '...'}
    node_addr_orig = client_info.get("address")
    # print(f\"FL Client {client_id} ({node_addr_orig}): Loading real timeseries data\")
    
    if not TS_EVENT_CACHE["processed"]:
        fetch_and_cache_timeseries_events(bc_connector)

    client_input_seqs = []
    client_target_seqs = [] 

    try:
        node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr_orig)
        input_s, target_s = generate_timeseries_for_node(node_addr_checksum, TS_EVENT_CACHE, bc_connector.w3, sybil_attack_log)
        client_input_seqs.append(input_s)
        client_target_seqs.append(target_s)
    except Exception as e:
        print(f"Error processing node {node_addr_orig} for client {client_id} timeseries: {e}")
        import traceback
        print(traceback.format_exc())

    if not client_input_seqs:
        return tf.data.Dataset.from_tensor_slices((\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)\
        )).batch(1)

    input_array = np.array(client_input_seqs, dtype=np.float32)
    target_array = np.array(client_target_seqs, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((input_array, target_array))
    # Each client gets data for one node, so batch size 1 is appropriate here.
    # Shuffling is not strictly necessary if each client has only one timeseries sample.
    return dataset.batch(1) 


def make_federated_data_p3_timeseries_real(clients_info: list[dict], sybil_attack_log=None):
    # clients_info: list of dicts, each like {'address': '0x...', 'role': '...'}
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded (Timeseries).")
        empty_ds = tf.data.Dataset.from_tensor_slices((\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)\
        )).batch(1)
        return [empty_ds for _ in range(len(clients_info))]

    fetch_and_cache_timeseries_events(bc_connector)
    if not TS_EVENT_CACHE["processed"]:
        print("Critical Error: Failed to fetch timeseries events (Timeseries).")
        empty_ds = tf.data.Dataset.from_tensor_slices((\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),\
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)\
        )).batch(1)
        return [empty_ds for _ in range(len(clients_info))]

    if not clients_info:
        print("Warning: No client info provided for timeseries data. Returning empty list.")
        return []

    client_datasets = []
    for i, client_data in enumerate(clients_info):
        client_id = f"real_ts_client_{i}_{client_data.get('address', 'unknown')[:8]}"
        client_ds = load_real_data_for_fl_client_timeseries(client_id, client_data, bc_connector, sybil_attack_log)
        client_datasets.append(client_ds)
    
    print(f"Created {len(client_datasets)} client datasets for timeseries analysis.")
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Node Behavior Timeseries (Autoencoder Data with Sybil Log)...")
    
    # Mock clients_info (list of dicts with address and role)
    mock_clients_info = [
        {"address": "0x0000000000000000000000000000000000000001", "role": "Manufacturer"},
        {"address": "0xSyBiL00000000000000000000000000000000001", "role": "SybilNode_Sybil1"},
        {"address": "0xBribedNode000000000000000000000000000001", "role": "Bribed_Retailer"}
    ]
    if not mock_clients_info:
        print("Warning: mock_clients_info is empty. Add mock client data for testing.")
        sys.exit(1)

    print(f"Using mock client info: {mock_clients_info}")

    # Create a mock sybil_attack_log.json content
    mock_sybil_log_content = {
        "simulationDate": "2023-10-27T12:00:00.000Z",
        "simulationEndTime": "2023-10-27T14:00:00.000Z",
        "sybilNodes": [
            {"id": "Sybil1", "address": "0xSyBiL00000000000000000000000000000000001"}
        ],
        "scenarioA": { # Frivolous disputes by Sybil1
            "actions": [
                {"type": "initiateFrivolousDispute", "sybilNode": "0xSyBiL00000000000000000000000000000000001", "status": "success", "timestamp": "2023-10-27T13:00:00.000Z"}
            ]
        },
        "scenarioB": {}, # Collusive voting
        "scenarioC": { # Counterfeit batch
            "actions": [
                {"type": "proposeCounterfeitBatch", "proposerNode": "0xSyBiL00000000000000000000000000000000001", "status": "success", "timestamp": "2023-10-27T13:05:00.000Z"}
            ]
        },
        "scenarioD": { # Bribery
            "bribedNodes": [
                {"address": "0xBribedNode0000000000000000000000000000001", "role": "Retailer", "timestamp": "2023-10-27T12:30:00.000Z"}
            ],
            "actions": [
                {"type": "bribePaymentAttempt", "targetAddress": "0xBribedNode0000000000000000000000000000001", "status": "success", "timestamp": "2023-10-27T12:30:00.000Z"}
            ],
            "simulatedBehavioralChanges": [
                {"actor": "0xBribedNode0000000000000000000000000000001", "details": ["Expected to act maliciously"], "timestamp": "2023-10-27T12:35:00.000Z"}
            ]
        }
    }

    federated_train_data_test = make_federated_data_p3_timeseries_real(
        clients_info=mock_clients_info, 
        sybil_attack_log=mock_sybil_log_content
    )
    
    print(f"\\nCreated {len(federated_train_data_test)} client datasets for timeseries.")
    for i, client_dataset in enumerate(federated_train_data_test):
        client_address_for_log = mock_clients_info[i]["address"] if i < len(mock_clients_info) else "Unknown"
        print(f"Client {i} (Address: {client_address_for_log}) dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        for input_batch, target_batch in client_dataset.take(1): # Each client dataset should have 1 element (1 node's timeseries)
            for j_idx in range(input_batch.shape[0]): 
                num_elements += 1
                print(f"  Client {i}, Sample {num_elements}: Input Seq Shape: {input_batch[j_idx].shape}, Target Seq Shape: {target_batch[j_idx].shape}")
                # Check the malicious flag for a few timesteps
                print(f"    Example - Malicious Flag (last 5 steps): {input_batch[j_idx].numpy()[-5:, 3]}")
                # print(f"    Input example (first 5 steps, all features): \n{input_batch[j_idx].numpy()[:5]}") 
        if num_elements == 0: print(f"  Client {i} (Address: {client_address_for_log}) has no data.")
        
        # total_samples_in_client = sum(tf.shape(batch[0])[0].numpy() for batch in client_dataset)
        # print(f"  Client {i} total samples: {total_samples_in_client}") # Should be 1 per client

    print("\\nReal data preparation test for Node Behavior Timeseries with Sybil log complete.")

