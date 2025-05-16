# Real Data Preparation for Node Behavior Timeseries FL Model (Phase 3)

import tensorflow as tf
import numpy as np
import time
import sys
from collections import defaultdict, deque

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/")

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_TS_FEATURES = 3  # tx_frequency, tx_value_sum, new_counterparties
TIMESTEPS = 30  # e.g., 30 days or 30 time periods
PERIOD_DURATION_SECONDS = 86400  # Duration of each timestep in seconds (e.g., 1 day)

ELEMENT_SPEC_P3_TS = (
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32),
    # For an autoencoder, the "label" is the input sequence itself, aiming for reconstruction.
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

def generate_timeseries_for_node(node_address_checksum, events_cache, w3_instance):
    sequence = np.zeros((TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
    if not events_cache["processed"] or not events_cache.get("all_events_sorted"):
        # print(f"Event cache not processed for node {node_address_checksum}. Returning zero sequence.")
        return sequence, sequence 

    node_start_time = get_node_start_timestamp(node_address_checksum, events_cache)
    current_processing_time = int(time.time())
    end_of_last_period = current_processing_time
    known_counterparties_for_node = set()

    for i in range(TIMESTEPS):
        period_end_time = end_of_last_period - (i * PERIOD_DURATION_SECONDS)
        period_start_time = period_end_time - PERIOD_DURATION_SECONDS
        if period_start_time < node_start_time - PERIOD_DURATION_SECONDS: 
            continue

        tx_frequency = 0
        tx_value_sum = 0.0
        current_period_counterparties = set()

        for event in events_cache["all_events_sorted"]:
            event_ts = event.get("timestamp", 0)
            if not (period_start_time <= event_ts < period_end_time):
                continue

            is_node_involved = False
            event_value = 0.0
            counterparty_in_event = None
            args = event.get("args", {})

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
                # Consider if being the defendant also counts as an "activity" for this node
                elif args.get("currentOwner") == node_address_checksum: 
                    is_node_involved = True; counterparty_in_event = args.get("initiator")
            elif event["event"] == "BatchProposed" and args.get("proposer") == node_address_checksum:
                is_node_involved = True 
            elif event["event"] == "BatchValidated" and args.get("validator") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "ArbitratorVoted" and args.get("voter") == node_address_checksum:
                is_node_involved = True
            elif event["event"] == "NodeVerified" and args.get("node") == node_address_checksum:
                 is_node_involved = True # Verification itself is an event

            if is_node_involved:
                tx_frequency += 1
                tx_value_sum += event_value
                if counterparty_in_event and isinstance(counterparty_in_event, str) and w3_instance.to_checksum_address(counterparty_in_event) != node_address_checksum:
                    current_period_counterparties.add(w3_instance.to_checksum_address(counterparty_in_event))
        
        new_counterparties_this_period = len(current_period_counterparties - known_counterparties_for_node)
        known_counterparties_for_node.update(current_period_counterparties)
        
        sequence_idx = TIMESTEPS - 1 - i
        sequence[sequence_idx, 0] = float(tx_frequency)
        sequence[sequence_idx, 1] = float(tx_value_sum) 
        sequence[sequence_idx, 2] = float(new_counterparties_this_period)

    # Normalize features (example: per-feature standardization or min-max scaling based on global stats if available)
    # For now, using raw values. Normalization can be added as a separate step or within the TFF model.
    return sequence, sequence 

def load_real_data_for_fl_client_timeseries(client_id: str, target_node_addresses: list[str], bc_connector):
    # print(f"FL Client {client_id}: Loading real timeseries data for nodes: {target_node_addresses}")
    if not TS_EVENT_CACHE["processed"]:
        fetch_and_cache_timeseries_events(bc_connector)

    client_input_seqs = []
    client_target_seqs = [] 

    for node_addr_orig in target_node_addresses:
        try:
            node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr_orig)
            # print(f"  Client {client_id}: Processing node {node_addr_checksum} for timeseries")
            input_s, target_s = generate_timeseries_for_node(node_addr_checksum, TS_EVENT_CACHE, bc_connector.w3)
            client_input_seqs.append(input_s)
            client_target_seqs.append(target_s)
            time.sleep(0.01) 
        except Exception as e:
            print(f"Error processing node {node_addr_orig} for client {client_id} timeseries: {e}")

    if not client_input_seqs:
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        )).batch(1)

    input_array = np.array(client_input_seqs, dtype=np.float32)
    target_array = np.array(client_target_seqs, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((input_array, target_array))
    return dataset.shuffle(200).batch(16)

def make_federated_data_p3_timeseries_real(all_target_node_addresses: list[str], num_fl_clients: int):
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded (Timeseries).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    fetch_and_cache_timeseries_events(bc_connector)
    if not TS_EVENT_CACHE["processed"]:
        print("Critical Error: Failed to fetch timeseries events (Timeseries).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if not all_target_node_addresses:
        print("Warning: No target node addresses provided for timeseries data. Will attempt to use all nodes with any activity if possible, or return empty.")
        # Fallback: try to get all nodes that had any activity if list is empty
        # This could be very broad. For now, if empty, return empty datasets.
        if not TS_EVENT_CACHE.get("all_events_sorted"):
             empty_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
                np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
            )).batch(1)
             return [empty_ds for _ in range(num_fl_clients)]
        # A more sophisticated approach might be to discover all unique addresses from events.
        # For now, require explicit node list if all_target_node_addresses is empty.
        print("No target nodes specified and auto-discovery not implemented for timeseries. Returning empty datasets.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]


    nodes_per_client = len(all_target_node_addresses) // num_fl_clients
    extra_nodes = len(all_target_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_ts_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(all_target_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(all_target_node_addresses))
            client_node_list = all_target_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list:
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
                np.empty((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client_timeseries(client_id, client_node_list, bc_connector)
        client_datasets.append(client_ds)
    
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Node Behavior Timeseries (Autoencoder Data)...")
    example_node_addresses = [
        # Add actual node addresses from your Amoy testnet deployment
        # "0xNodeAddress1", "0xNodeAddress2"
    ]
    if not example_node_addresses:
        print("Warning: example_node_addresses is empty. Using placeholders for structural test.")
        example_node_addresses.extend(["0x0000000000000000000000000000000000000001", "0x0000000000000000000000000000000000000002"])

    print(f"Using example node addresses: {example_node_addresses}")

    num_clients_test = 1
    federated_train_data_test = make_federated_data_p3_timeseries_real(example_node_addresses, num_fl_clients=num_clients_test)
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets for timeseries.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        for input_batch, target_batch in client_dataset.take(1): 
            for j_idx in range(input_batch.shape[0]): 
                num_elements += 1
                print(f"  Client {i}, Sample {num_elements}: Input Seq Shape: {input_batch[j_idx].shape}, Target Seq Shape: {target_batch[j_idx].shape}")
                # print(f"    Input example (first 5 steps): {input_batch[j_idx].numpy()[:5]}") 
        if num_elements == 0: print(f"  Client {i} has no data.")
        
        total_samples_in_client = sum(tf.shape(batch[0])[0].numpy() for batch in client_dataset)
        print(f"  Client {i} total samples: {total_samples_in_client}")

    print("\nReal data preparation test for Node Behavior Timeseries complete.")
    print("The model will learn to reconstruct these sequences. Anomalies are detected by high reconstruction error post-training.")

