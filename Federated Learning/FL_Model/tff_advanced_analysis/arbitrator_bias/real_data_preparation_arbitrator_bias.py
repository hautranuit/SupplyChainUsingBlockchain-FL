# Real Data Preparation for Arbitrator Bias FL Model (Phase 3)

import tensorflow as tf
import numpy as np
import time
import sys
from collections import defaultdict, Counter

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/")

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_ARB_FEATURES = 5
ELEMENT_SPEC_P3_ARB = (
    tf.TensorSpec(shape=(NUM_P3_ARB_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for not biased, 1 for biased
)

RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

DISPUTE_EVENT_CACHE = {
    "DisputeInitiated": None,
    "ArbitratorSelected": None,
    "ArbitratorVoted": None,
    "DisputeResolved": None,
    "processed": False
}

# Heuristic thresholds for Arbitrator Bias labeling (can be tuned)
MIN_DISPUTES_FOR_BIAS_EVAL = 5 # Minimum disputes an arbitrator must have participated in to be evaluated for bias
ALIGNMENT_OUTCOME_THRESHOLD_LOW = 0.4 # If alignment with outcome is below this, potentially biased
ALIGNMENT_PEER_THRESHOLD_LOW = 0.4    # If alignment with peer majority is below this, potentially biased
PARTY_VOTE_SKEW_THRESHOLD = 0.8       # If votes for one party (initiator/defendant) exceed this percentage of total votes
                                      # (and the other party is below 1 - SKEW_THRESHOLD), indicates strong preference.

def fetch_and_cache_dispute_events(bc_connector):
    if DISPUTE_EVENT_CACHE["processed"]:
        return
    print("Fetching and caching dispute processing events for Arbitrator Bias...")
    try:
        DISPUTE_EVENT_CACHE["DisputeInitiated"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="DisputeInitiated", from_block=0
        )
        DISPUTE_EVENT_CACHE["ArbitratorSelected"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="ArbitratorSelected", from_block=0
        )
        DISPUTE_EVENT_CACHE["ArbitratorVoted"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="ArbitratorVoted", from_block=0
        )
        DISPUTE_EVENT_CACHE["DisputeResolved"] = bc_connector.get_events(
            contract_name="SupplyChainNFT", event_name="DisputeResolved", from_block=0
        )
        DISPUTE_EVENT_CACHE["processed"] = True
        # print(f"Fetched {len(DISPUTE_EVENT_CACHE["DisputeInitiated"])} DisputeInitiated events.")
    except Exception as e:
        print(f"Error fetching dispute events for Arbitrator Bias: {e}")
        DISPUTE_EVENT_CACHE["processed"] = False
        for key in DISPUTE_EVENT_CACHE: DISPUTE_EVENT_CACHE[key] = [] if key != "processed" else False

def generate_features_and_label_for_arbitrator(arbitrator_address_checksum, all_events, w3_instance):
    features = np.zeros(NUM_P3_ARB_FEATURES, dtype=np.float32)
    bias_score = 0

    arbitrator_participated_token_ids = set()
    if all_events.get("ArbitratorSelected"):
        for event in all_events["ArbitratorSelected"]:
            if event["args"]["arbitrator"] == arbitrator_address_checksum:
                arbitrator_participated_token_ids.add(event["args"]["tokenId"])
    
    disputes_participated_count = len(arbitrator_participated_token_ids)
    features[2] = float(disputes_participated_count) # F2: disputes_participated_count

    if disputes_participated_count < MIN_DISPUTES_FOR_BIAS_EVAL:
        label = np.array([0], dtype=np.int32) # Not enough data to label as biased
        return features, label

    votes_aligned_with_outcome = 0
    votes_aligned_with_peer_majority = 0
    votes_for_initiator = 0
    votes_for_defendant = 0 # Defendant is currentOwner in DisputeInitiated
    total_votes_cast_by_arbitrator = 0

    for token_id in arbitrator_participated_token_ids:
        my_vote_event = next((v for v in all_events.get("ArbitratorVoted", []) 
                              if v["args"]["tokenId"] == token_id and 
                                 v["args"]["voter"] == arbitrator_address_checksum), None)
        if not my_vote_event:
            continue
        
        total_votes_cast_by_arbitrator += 1
        my_voted_candidate = my_vote_event["args"]["candidate"]

        dispute_initiated_event = next((di for di in all_events.get("DisputeInitiated", []) 
                                        if di["args"]["tokenId"] == token_id), None)
        if not dispute_initiated_event:
            continue
        initiator = dispute_initiated_event["args"]["initiator"]
        defendant = dispute_initiated_event["args"]["currentOwner"]

        if my_voted_candidate == initiator:
            votes_for_initiator += 1
        elif my_voted_candidate == defendant:
            votes_for_defendant += 1

        resolved_event = next((r for r in all_events.get("DisputeResolved", []) 
                               if r["args"]["tokenId"] == token_id), None)
        if resolved_event:
            outcome_favors_initiator = resolved_event["args"]["result"]
            if (outcome_favors_initiator and my_voted_candidate == initiator) or \
               (not outcome_favors_initiator and my_voted_candidate == defendant):
                votes_aligned_with_outcome += 1
        
        peer_votes_for_token = [v["args"]["candidate"] for v in all_events.get("ArbitratorVoted", []) 
                                if v["args"]["tokenId"] == token_id and 
                                   v["args"]["voter"] != arbitrator_address_checksum]
        if peer_votes_for_token:
            majority_candidate_counter = Counter(peer_votes_for_token)
            if majority_candidate_counter:
                max_peer_votes = max(majority_candidate_counter.values())
                majority_candidates = [cand for cand, count in majority_candidate_counter.items() if count == max_peer_votes]
                if my_voted_candidate in majority_candidates:
                    votes_aligned_with_peer_majority += 1
    
    if total_votes_cast_by_arbitrator > 0:
        alignment_outcome_rate = float(votes_aligned_with_outcome / total_votes_cast_by_arbitrator)
        alignment_peer_rate = float(votes_aligned_with_peer_majority / total_votes_cast_by_arbitrator)
        initiator_vote_rate = float(votes_for_initiator / total_votes_cast_by_arbitrator)
        defendant_vote_rate = float(votes_for_defendant / total_votes_cast_by_arbitrator)

        features[0] = alignment_outcome_rate    # F0: vote_alignment_with_outcome
        features[1] = alignment_peer_rate       # F1: vote_alignment_with_peers
        features[3] = initiator_vote_rate       # F3: proportion_votes_for_initiator
        features[4] = defendant_vote_rate       # F4: proportion_votes_for_defendant

        # Heuristic Labeling for Bias
        if alignment_outcome_rate < ALIGNMENT_OUTCOME_THRESHOLD_LOW:
            bias_score += 1
        if alignment_peer_rate < ALIGNMENT_PEER_THRESHOLD_LOW:
            bias_score += 1
        if (initiator_vote_rate > PARTY_VOTE_SKEW_THRESHOLD and defendant_vote_rate < (1.0 - PARTY_VOTE_SKEW_THRESHOLD + 0.1)) or \
           (defendant_vote_rate > PARTY_VOTE_SKEW_THRESHOLD and initiator_vote_rate < (1.0 - PARTY_VOTE_SKEW_THRESHOLD + 0.1)):
            # The +0.1 is to ensure there's a clear skew, not just e.g. 0.8 vs 0.2 which could be normal if one party is often right
            # This checks for strong preference for one side over the other consistently.
            bias_score += 2 # Stronger indicator of bias
    
    label_val = 1 if bias_score >= 2 else 0 # Threshold for labeling as biased
    label = np.array([label_val], dtype=np.int32)

    if label_val == 1:
        print(f"  Arbitrator {arbitrator_address_checksum} labeled BIASED (score: {bias_score}). Participated: {disputes_participated_count}, AlignOutcome: {features[0]:.2f}, AlignPeer: {features[1]:.2f}, VoteInit: {features[3]:.2f}, VoteDef: {features[4]:.2f}")
    return features, label

def load_real_data_for_fl_client_arbitrator_bias(client_id: str, target_node_addresses: list[str], bc_connector):
    # print(f"FL Client {client_id}: Loading real arbitrator bias data for nodes: {target_node_addresses}")
    if not DISPUTE_EVENT_CACHE["processed"]:
        fetch_and_cache_dispute_events(bc_connector)
    
    if not DISPUTE_EVENT_CACHE["processed"] or not any(DISPUTE_EVENT_CACHE.get(k) for k in ["ArbitratorSelected", "ArbitratorVoted"]):
        print(f"Warning: Dispute event cache is empty or incomplete for client {client_id} (Arbitrator Bias).")

    client_features_list = []
    client_labels_list = []

    for node_addr_orig in target_node_addresses:
        try:
            node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr_orig)
            # print(f"  Client {client_id}: Processing arbitrator {node_addr_checksum} for bias")
            features, label = generate_features_and_label_for_arbitrator(node_addr_checksum, DISPUTE_EVENT_CACHE, bc_connector.w3)
            client_features_list.append(features)
            client_labels_list.append(label)
            time.sleep(0.01) # Reduced delay as event fetching is cached
        except Exception as e:
            print(f"Error processing arbitrator {node_addr_orig} for client {client_id} (Arbitrator Bias): {e}")

    if not client_features_list:
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    return dataset.shuffle(1000).batch(32)

def make_federated_data_p3_arbitrator_real(all_potential_arbitrator_addresses: list[str], num_fl_clients: int):
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded (Arbitrator Bias).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    fetch_and_cache_dispute_events(bc_connector) # Ensure cache is populated before filtering
    if not DISPUTE_EVENT_CACHE["processed"]:
        print("Critical Error: Failed to fetch dispute events. Cannot generate arbitrator data.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    actual_arbitrators = set()
    if DISPUTE_EVENT_CACHE.get("ArbitratorSelected"):
        for event in DISPUTE_EVENT_CACHE["ArbitratorSelected"]:
            actual_arbitrators.add(event["args"]["arbitrator"])
    
    # Use all actual arbitrators found in events if no specific list is provided or if the list is empty
    if not all_potential_arbitrator_addresses:
        print("No potential arbitrator addresses provided, using all found in ArbitratorSelected events.")
        target_node_addresses = list(actual_arbitrators)
    else:
        target_node_addresses = [addr for addr in all_potential_arbitrator_addresses if bc_connector.w3.to_checksum_address(addr) in actual_arbitrators]

    if not target_node_addresses:
        print("Warning: No target arbitrators found (either from input list or events). Returning empty datasets for Arbitrator Bias.")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]
    
    print(f"Found {len(target_node_addresses)} actual arbitrators to process for bias detection.")

    nodes_per_client = len(target_node_addresses) // num_fl_clients
    extra_nodes = len(target_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_arb_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(target_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(target_node_addresses))
            client_node_list = target_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list:
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client_arbitrator_bias(client_id, client_node_list, bc_connector)
        client_datasets.append(client_ds)
    
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Arbitrator Bias with Heuristic Labeling...")
    example_arbitrator_addresses = [
        # Add actual arbitrator addresses from Amoy testnet for a real test
        # e.g., from your dApp interactions or known participants
    ]
    if not example_arbitrator_addresses:
        print("Warning: example_arbitrator_addresses is empty. Test will rely on arbitrators found in events.")
        # If you want to test specific addresses, add them here. Otherwise, it will use all found arbitrators.
        # example_arbitrator_addresses.append("0x0000000000000000000000000000000000000001") # Dummy for structure if needed

    num_clients_test = 1
    federated_train_data_test = make_federated_data_p3_arbitrator_real(example_arbitrator_addresses, num_fl_clients=num_clients_test)
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets for arbitrator bias.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        biased_count = 0
        normal_count = 0
        for features_batch, labels_batch in client_dataset:
            for j_idx in range(features_batch.shape[0]):
                num_elements += 1
                # print(f"  Client {i}, Sample {num_elements}: Features: {features_batch[j_idx].numpy()}, Label: {labels_batch[j_idx].numpy()}")
                if labels_batch[j_idx].numpy()[0] == 1:
                    biased_count +=1
                else:
                    normal_count +=1
        if num_elements == 0: print(f"  Client {i} has no data.")
        print(f"  Client {i} total samples: {num_elements}. Biased: {biased_count}, Normal: {normal_count}")

    print("\nReal data preparation test for Arbitrator Bias with labeling complete.")

