# Real Data Preparation for Dispute Risk FL Model (Phase 3)

import tensorflow as tf
import numpy as np
import time
import sys
from collections import defaultdict

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/")

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF
NUM_P3_DR_FEATURES = 6 
ELEMENT_SPEC_P3_DR = (
    tf.TensorSpec(shape=(NUM_P3_DR_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for low risk, 1 for high risk
)

RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

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

def fetch_and_cache_dispute_risk_events(bc_connector):
    if DISPUTE_EVENT_CACHE_DR["processed"]:
        return
    print("Fetching and caching events for dispute risk analysis...")
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
        # print(f"Fetched {len(DISPUTE_EVENT_CACHE_DR["DisputeInitiated"])} DisputeInitiated events for DR.")
    except Exception as e:
        print(f"Error fetching dispute risk events: {e}")
        DISPUTE_EVENT_CACHE_DR["processed"] = False
        for key in DISPUTE_EVENT_CACHE_DR: DISPUTE_EVENT_CACHE_DR[key] = [] if key != "processed" else False

def get_dispute_value(token_id, events_cache):
    collateral_event = next((e for e in reversed(events_cache.get("CollateralDepositedForPurchase", [])) if e["args"]["tokenId"] == token_id), None)
    if collateral_event:
        return float(collateral_event["args"]["amount"])
    listing_event = next((e for e in reversed(events_cache.get("ProductListedForSale", [])) if e["args"]["tokenId"] == token_id), None)
    if listing_event:
        return float(listing_event["args"]["price"])
    return 0.0

def get_node_age_seconds(node_address_checksum, events_cache, current_timestamp, w3_instance):
    verified_events = [e for e in events_cache.get("NodeVerified", []) if e["args"]["node"] == node_address_checksum]
    if not verified_events:
        return 0
    verified_events.sort(key=lambda x: x["blockNumber"])
    first_verification_event = verified_events[0]
    block_number = first_verification_event["blockNumber"]
    try:
        timestamp = w3_instance.eth.get_block(block_number)["timestamp"]
        return float(current_timestamp - timestamp)
    except Exception:
        return 0

def generate_features_and_label_for_dispute(dispute_event, bc_connector, events_cache):
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

    # F1: Number of Parties (currently fixed at 2)
    features[1] = 2.0 

    # F2: Average Reputation of Parties
    rep_initiator = bc_connector.get_node_reputation(initiator_checksum) or 0
    rep_defendant = bc_connector.get_node_reputation(defendant_checksum) or 0
    avg_reputation = (float(rep_initiator) + float(rep_defendant)) / 2.0
    features[2] = avg_reputation
    if avg_reputation < LOW_AVERAGE_REPUTATION_THRESHOLD:
        risk_score += 1

    # F3: Reputation Difference
    reputation_diff = abs(float(rep_initiator) - float(rep_defendant))
    features[3] = reputation_diff
    if reputation_diff > LARGE_REPUTATION_DIFFERENCE_THRESHOLD:
        risk_score += 1

    # F4: Evidence Count
    evidence_count = sum(1 for e in events_cache.get("CIDStored", []) if e["args"]["tokenId"] == token_id)
    features[4] = float(evidence_count)
    if evidence_count < LOW_EVIDENCE_COUNT_THRESHOLD:
        risk_score += 1

    # F5: Initiator Prior Risk (based on age)
    initiator_age_seconds = get_node_age_seconds(initiator_checksum, events_cache, current_time, bc_connector.w3)
    initiator_prior_risk_feature_val = 1.0 # Max risk if age is 0 or very new
    if initiator_age_seconds > 0:
        # Normalize age to months, then inverse+1. Max value is 1 (for age=0), decreases as age increases.
        initiator_prior_risk_feature_val = 1.0 / ( (initiator_age_seconds / (86400 * 30)) + 1.0)
    features[5] = initiator_prior_risk_feature_val
    
    if initiator_prior_risk_feature_val > INITIATOR_HIGH_PRIOR_RISK_THRESHOLD: # Higher value means newer node
        risk_score += 2
    elif initiator_age_seconds == 0: # If age is truly zero (couldn't be found), also high risk
        risk_score += 2
    
    # Additional check: if initiator is very new (e.g. < 2 weeks), it might be a risky dispute
    if 0 < initiator_age_seconds < MIN_INITIATOR_AGE_FOR_LOW_RISK_SECONDS:
        risk_score +=1

    label_val = 1 if risk_score >= 3 else 0 # Threshold for labeling as high-risk dispute
    label = np.array([label_val], dtype=np.int32)

    if label_val == 1:
        print(f"  Dispute for token {token_id} labeled HIGH RISK (score: {risk_score}). Val:{features[0]:.0f}, AvgRep:{features[2]:.1f}, RepDiff:{features[3]:.1f}, Evid:{features[4]:.0f}, InitRisk:{features[5]:.2f}")
    return features, label

def load_real_data_for_fl_client_dispute_risk(client_id: str, dispute_events_for_client: list, bc_connector, events_cache):
    # print(f"FL Client {client_id}: Loading real dispute risk data for {len(dispute_events_for_client)} disputes")
    client_features_list = []
    client_labels_list = []

    for dispute_event in dispute_events_for_client:
        try:
            features, label = generate_features_and_label_for_dispute(dispute_event, bc_connector, events_cache)
            client_features_list.append(features)
            client_labels_list.append(label)
            time.sleep(0.01) 
        except Exception as e:
            print(f"Error processing dispute event for client {client_id} (Dispute Risk): {e}")

    if not client_features_list:
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    return dataset.shuffle(500).batch(32)

def make_federated_data_p3_dispute_real(num_fl_clients: int):
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    fetch_and_cache_dispute_risk_events(bc_connector)
    if not DISPUTE_EVENT_CACHE_DR["processed"] or not DISPUTE_EVENT_CACHE_DR.get("DisputeInitiated"):
        print("Critical Error: Failed to fetch DisputeInitiated events (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    all_dispute_events = DISPUTE_EVENT_CACHE_DR["DisputeInitiated"]
    if not all_dispute_events:
        print("Warning: No DisputeInitiated events found in cache (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]
    
    np.random.shuffle(all_dispute_events)
    disputes_per_client = len(all_dispute_events) // num_fl_clients
    extra_disputes = len(all_dispute_events) % num_fl_clients
    client_datasets = []
    current_dispute_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_dr_client_{i}"
        num_disputes_for_this_client = disputes_per_client + (1 if i < extra_disputes else 0)
        client_dispute_list = []
        if num_disputes_for_this_client > 0 and current_dispute_idx < len(all_dispute_events):
            end_idx = min(current_dispute_idx + num_disputes_for_this_client, len(all_dispute_events))
            client_dispute_list = all_dispute_events[current_dispute_idx : end_idx]
            current_dispute_idx = end_idx
        
        if not client_dispute_list:
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client_dispute_risk(client_id, client_dispute_list, bc_connector, DISPUTE_EVENT_CACHE_DR)
        client_datasets.append(client_ds)
    
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Dispute Risk with Heuristic Labeling...")
    num_clients_test = 1 
    
    federated_train_data_test = make_federated_data_p3_dispute_real(num_fl_clients=num_clients_test)
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets for dispute risk.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        high_risk_count = 0
        low_risk_count = 0
        for features_batch, labels_batch in client_dataset:
            for j_idx in range(features_batch.shape[0]):
                num_elements += 1
                # print(f"  Client {i}, Sample {num_elements}: Features: {features_batch[j_idx].numpy()}, Label: {labels_batch[j_idx].numpy()}")
                if labels_batch[j_idx].numpy()[0] == 1:
                    high_risk_count +=1
                else:
                    low_risk_count +=1
        if num_elements == 0: print(f"  Client {i} has no data (or no disputes found).")
        print(f"  Client {i} total samples: {num_elements}. High-Risk: {high_risk_count}, Low-Risk: {low_risk_count}")

    print("\nReal data preparation test for Dispute Risk with labeling complete.")

