# Real Data Preparation for Dispute Risk FL Model (Phase 3)

import tensorflow as tf
import numpy as np
import time
import sys
import json # Added for loading sybil_attack_log
from collections import defaultdict

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/") # Keep existing sys.path
# If running locally and the above path is not correct, you might need to adjust it, e.g.:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \'../../..\')))


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
SYBIL_DISPUTE_RISK_INCREASE = 3 # Additional risk score for disputes initiated by known Sybils/malicious actors

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
        # print(f\"Fetched {len(DISPUTE_EVENT_CACHE_DR[\"DisputeInitiated\"])} DisputeInitiated events for DR.\")
    except Exception as e:
        print(f\"Error fetching dispute risk events: {e}\")
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

def generate_features_and_label_for_dispute(dispute_event, bc_connector, events_cache, sybil_attack_log=None):
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

    # Incorporate Sybil attack log information
    is_malicious_dispute = False
    if sybil_attack_log:
        # Check Scenario A: Frivolous disputes by Sybils
        if 'scenarioA' in sybil_attack_log and 'actions' in sybil_attack_log['scenarioA']:
            for action in sybil_attack_log['scenarioA']['actions']:
                if action.get('type') == 'initiateFrivolousDispute' and \
                   action.get('status') == 'success' and \
                   bc_connector.w3.to_checksum_address(action.get('sybilNode')) == initiator_checksum and \
                   action.get('tokenId') == token_id:
                    risk_score += SYBIL_DISPUTE_RISK_INCREASE
                    is_malicious_dispute = True
                    print(f"  Sybil Check: Dispute for token {token_id} by {initiator_checksum} (Scenario A) identified. Risk increased.")
                    break
        
        # Check Scenario D: Disputes potentially initiated by bribed nodes (if applicable to dispute creation)
        # This part is more speculative as Scenario D focuses on bribe payments, not direct dispute creation.
        # However, a bribed node *might* initiate disputes as part of its malicious behavior.
        # For now, we'll assume bribed nodes don't directly create disputes unless explicitly logged.
        # If specific dispute creation actions by bribed nodes are logged in scenarioD.actions or scenarioD.simulatedBehavioralChanges,
        # similar logic to Scenario A could be added here.
        if not is_malicious_dispute and 'scenarioD' in sybil_attack_log and 'bribedNodes' in sybil_attack_log['scenarioD']:
            for bribed_node_info in sybil_attack_log['scenarioD']['bribedNodes']:
                if bc_connector.w3.to_checksum_address(bribed_node_info.get('address')) == initiator_checksum:
                    # This indicates the initiator was bribed. We might infer increased risk for any dispute they raise.
                    # The `simulatedBehavioralChanges` could provide more specific clues.
                    for change in sybil_attack_log['scenarioD'].get('simulatedBehavioralChanges', []):
                        if bc_connector.w3.to_checksum_address(change.get('actor')) == initiator_checksum:
                            if any("dispute" in detail.lower() for detail in change.get('details',[])):
                                risk_score += SYBIL_DISPUTE_RISK_INCREASE # Or a different value for bribed actor disputes
                                is_malicious_dispute = True
                                print(f"  Sybil Check: Dispute for token {token_id} by bribed node {initiator_checksum} (Scenario D behavior) identified. Risk increased.")
                                break
                    if is_malicious_dispute: break


    label_val = 1 if risk_score >= 3 else 0 # Threshold for labeling as high-risk dispute
    if is_malicious_dispute and label_val == 0: # Ensure explicitly malicious disputes are labeled high risk
        label_val = 1
        print(f"  Sybil Override: Dispute for token {token_id} by {initiator_checksum} forced to HIGH RISK due to Sybil activity.")


    label = np.array([label_val], dtype=np.int32)

    if label_val == 1:
        print(f"  Dispute for token {token_id} labeled HIGH RISK (score: {risk_score}). Val:{features[0]:.0f}, AvgRep:{features[2]:.1f}, RepDiff:{features[3]:.1f}, Evid:{features[4]:.0f}, InitRisk:{features[5]:.2f}, Malicious: {is_malicious_dispute}")
    return features, label

def load_real_data_for_fl_client_dispute_risk(client_id: str, client_address: str, all_dispute_events: list, bc_connector, events_cache, sybil_attack_log=None):
    # print(f\"FL Client {client_id} ({client_address}): Loading real dispute risk data\")
    client_features_list = []
    client_labels_list = []

    # Filter disputes relevant to this client (initiator or defendant)
    client_specific_disputes = []
    for dispute_event in all_dispute_events:
        initiator_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["initiator"])
        defendant_checksum = bc_connector.w3.to_checksum_address(dispute_event["args"]["currentOwner"])
        if client_address == initiator_checksum or client_address == defendant_checksum:
            client_specific_disputes.append(dispute_event)
    
    # print(f\"  Client {client_id} has {len(client_specific_disputes)} relevant disputes out of {len(all_dispute_events)} total.\")

    for dispute_event in client_specific_disputes:
        try:
            features, label = generate_features_and_label_for_dispute(dispute_event, bc_connector, events_cache, sybil_attack_log)
            client_features_list.append(features)
            client_labels_list.append(label)
            time.sleep(0.01) 
        except Exception as e:
            print(f\"Error processing dispute event for client {client_id} (Dispute Risk): {e}\")

    if not client_features_list:
        return tf.data.Dataset.from_tensor_slices((\
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),\
            np.empty((0, 1), dtype=np.int32)\
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    return dataset.shuffle(len(client_features_list) if len(client_features_list) > 0 else 1).batch(32)


def make_federated_data_p3_dispute_real(all_node_addresses: list, sybil_attack_log=None):
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((\
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),\
            np.empty((0, 1), dtype=np.int32)\
        )).batch(1)
        return [empty_ds for _ in range(len(all_node_addresses))]

    fetch_and_cache_dispute_risk_events(bc_connector)
    if not DISPUTE_EVENT_CACHE_DR["processed"] or not DISPUTE_EVENT_CACHE_DR.get("DisputeInitiated"):
        print("Critical Error: Failed to fetch DisputeInitiated events (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((\
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),\
            np.empty((0, 1), dtype=np.int32)\
        )).batch(1)
        return [empty_ds for _ in range(len(all_node_addresses))]

    all_dispute_events = DISPUTE_EVENT_CACHE_DR["DisputeInitiated"]
    if not all_dispute_events:
        print("Warning: No DisputeInitiated events found in cache (Dispute Risk).")
        empty_ds = tf.data.Dataset.from_tensor_slices((\
            np.empty((0, NUM_P3_DR_FEATURES), dtype=np.float32),\
            np.empty((0, 1), dtype=np.int32)\
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
    
    print(f"Created {len(client_datasets)} client datasets for dispute risk based on {num_fl_clients} node addresses.")
    return client_datasets

# Add alias for backward compatibility
make_federated_data_dispute_risk_real = make_federated_data_p3_dispute_real

if __name__ == '__main__':
    print("Testing Real Data Preparation for Dispute Risk with Heuristic Labeling and Sybil Data...")
    
    # Mock all_node_addresses (replace with actual addresses if testing with a live environment)
    mock_node_addresses = [
        "0x0000000000000000000000000000000000000001", # Mock Client 1
        "0x0000000000000000000000000000000000000002", # Mock Client 2
        "0xSyBiL00000000000000000000000000000000001"  # Mock Sybil Node involved in a dispute
    ]
    # Ensure mock Sybil address is checksummed if it's used in mock_sybil_log
    # For simplicity, we'll assume addresses in mock_sybil_log are already as expected by the functions.

    # Create a mock sybil_attack_log.json content
    mock_sybil_log_content = {
        "simulationDate": "2023-10-27T10:00:00.000Z",
        "sybilNodes": [
            {"id": "Sybil1", "address": "0xSyBiL00000000000000000000000000000000001"}
        ],
        "scenarioA": {
            "description": "Frivolous disputes",
            "actions": [
                {
                    "type": "initiateFrivolousDispute",
                    "sybilNode": "0xSyBiL00000000000000000000000000000000001", # This should match a dispute initiator
                    "tokenId": 123, # This should match a tokenId in a DisputeInitiated event
                    "reason": "Fake reason",
                    "status": "success",
                    "txHash": "0xmocktxhash_frivolous_dispute"
                }
            ],
            "outcome": "Logged"
        },
        "scenarioD": {
            "description": "Bribery",
            "bribedNodes": [
                {"address": "0x0000000000000000000000000000000000000002", "role": "Transporter"} # A bribed node
            ],
            "simulatedBehavioralChanges": [
                {
                    "actor": "0x0000000000000000000000000000000000000002",
                    "details": ["Expected to initiate unfair disputes."] # Example detail
                }
            ],
            "actions": [],
            "outcome": "Logged"
        }
    }
    # To use this mock log, you would typically save it to a file and load it,
    # or pass the dictionary directly if your script structure allows.
    # For this test, we'll pass the dictionary directly.
    
    print(f"Using {len(mock_node_addresses)} mock FL clients.")
    
    # The make_federated_data function now expects all_node_addresses and sybil_attack_log
    federated_train_data_test = make_federated_data_p3_dispute_real(
        all_node_addresses=mock_node_addresses,
        sybil_attack_log=mock_sybil_log_content 
    )
    
    print(f"\\nCreated {len(federated_train_data_test)} client datasets for dispute risk.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} (Address: {mock_node_addresses[i] if i < len(mock_node_addresses) else \'N/A\'}) dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        high_risk_count = 0
        low_risk_count = 0
        for features_batch, labels_batch in client_dataset:
            for j_idx in range(features_batch.shape[0]):
                num_elements += 1
                # print(f\"  Client {i}, Sample {num_elements}: Features: {features_batch[j_idx].numpy()}, Label: {labels_batch[j_idx].numpy()}\")
                if labels_batch[j_idx].numpy()[0] == 1:
                    high_risk_count +=1
                else:
                    low_risk_count +=1
        if num_elements == 0: print(f"  Client {i} has no data (or no disputes found for this client).")
        else: print(f"  Client {i} total samples: {num_elements}. High-Risk: {high_risk_count}, Low-Risk: {low_risk_count}")

    print("\\nReal data preparation test for Dispute Risk with labeling and Sybil data complete.")

