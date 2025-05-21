# Real Data Preparation for Batch Monitoring FL Model (Phase 2)

import tensorflow as tf
import numpy as np
import time
import sys
import os
from collections import defaultdict
import json

# Adjust path to import connectors
sys.path.append("/home/ubuntu/fl_integration_workspace/Project/Federated Learning/FL_Model/")

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF, consistent with model_definition.py for Phase 2
NUM_BATCH_MONITORING_FEATURES = 6
ELEMENT_SPEC_BATCH_MONITORING = (
    tf.TensorSpec(shape=(NUM_BATCH_MONITORING_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for normal, 1 for anomalous
)

RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

EVENT_CACHE = {
    "BatchProposed": None,
    "BatchValidated": None,
    "BatchCommitted": None,
    "BatchStateChanged": None, # Added for counterfeit detection
    "processed": False
}

# Heuristic thresholds for Batch Monitoring labeling (can be tuned)
PROPOSER_MIN_SUCCESS_RATE = 0.6 # If a proposer's success rate is below this, potentially anomalous
PROPOSER_MAX_FAILED_BATCHES_ABSOLUTE = 5 # If a proposer has more than this many failed batches
VALIDATOR_MIN_AGREEMENT_RATE = 0.7 # If a validator's agreement with final outcome is below this
VALIDATOR_LOW_PARTICIPATION_THRESHOLD = 0.1 # If validator participates in <10% of batches they were selected for (harder to get from events directly, using total batches as proxy)

def fetch_and_cache_batch_events(bc_connector):
    if EVENT_CACHE["processed"]:
        return
    print("Fetching and caching batch processing events...")
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
        # Fetch BatchStateChanged events, assuming it's from BatchProcessing contract
        # Adjust contract_name if it's different
        EVENT_CACHE["BatchStateChanged"] = bc_connector.get_events(
            contract_name="BatchProcessing", event_name="BatchStateChanged", from_block=0
        )
        EVENT_CACHE["processed"] = True
        print(f"Fetched {len(EVENT_CACHE['BatchProposed'])} BatchProposed events.")
        print(f"Fetched {len(EVENT_CACHE['BatchValidated'])} BatchValidated events.")
        print(f"Fetched {len(EVENT_CACHE['BatchCommitted'])} BatchCommitted events.")
        print(f"Fetched {len(EVENT_CACHE['BatchStateChanged'])} BatchStateChanged events.")
    except Exception as e:
        print(f"Error fetching batch events: {e}")
        EVENT_CACHE["processed"] = False
        for key in EVENT_CACHE: EVENT_CACHE[key] = [] if key != "processed" else False

def generate_features_and_label_for_batch_monitoring(node_address_checksum, all_events):
    features = np.zeros(NUM_BATCH_MONITORING_FEATURES, dtype=np.float32)
    anomaly_score = 0

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
        if num_proposed_failed > PROPOSER_MAX_FAILED_BATCHES_ABSOLUTE:
            anomaly_score +=1

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
        
        # Rough participation rate check (could be improved if we know all batches a validator was *selected* for)
        total_batches_in_system = len(all_events.get("BatchProposed", []))
        if total_batches_in_system > 0 and (num_batches_validated_participation / total_batches_in_system) < VALIDATOR_LOW_PARTICIPATION_THRESHOLD:
            if num_batches_proposed == 0: # Only penalize if not also a proposer (proposers might not validate much)
                 anomaly_score += 1

    # Check for Scenario B (Collusive Voting) and Scenario C (Counterfeit Batch) from sybil_attack_log
    if is_known_sybil and sybil_attack_log and "attackDetails" in sybil_attack_log:
        for attack_entry in sybil_attack_log["attackDetails"]:
            # Scenario B: Collusive Voting
            if attack_entry.get("scenario") == "B" and attack_entry.get("action") == "Sybil nodes vote for batch" \
               and node_address_checksum in attack_entry.get("sybilsInvolved", []):
                batch_id_attack = attack_entry.get("batchId")
                # Check if this node actually voted on this batch according to blockchain events
                corresponding_vote_event = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id_attack), None)
                if corresponding_vote_event:
                    num_collusive_votes += 1
                    anomaly_score += 2 # Strong indicator
                    print(f"  Node {node_address_checksum} detected in collusive voting (Scenario B) for batch {batch_id_attack}")
            
            # Scenario C: Proposing/Approving Counterfeit Batch
            if attack_entry.get("scenario") == "C":
                if attack_entry.get("action") == "Sybil proposes counterfeit batch" and attack_entry.get("proposer") == node_address_checksum:
                    anomaly_score += 3 # Very strong indicator
                    print(f"  Node {node_address_checksum} detected proposing counterfeit batch (Scenario C) for batch {attack_entry.get('batchId')}")
                elif attack_entry.get("action") == "Sybils approve counterfeit batch" and node_address_checksum in attack_entry.get("sybilsInvolved", []):
                    batch_id_attack_c = attack_entry.get("batchId")
                    corresponding_vote_event_c = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id_attack_c), None)
                    if corresponding_vote_event_c and corresponding_vote_event_c["args"]["approve"]:
                        anomaly_score += 2 # Strong indicator for approving
                        print(f"  Node {node_address_checksum} detected approving counterfeit batch (Scenario C) for batch {batch_id_attack_c}")

    # Update features (placeholder for new features, e.g., collusive vote count)
    # features[X] = float(num_collusive_votes) # Example, if a new feature slot is added
    
    # If node is neither proposer nor validator in any significant way
    if num_batches_proposed == 0 and num_batches_validated_participation == 0 and len(all_events.get("BatchProposed",[])) > 5 : # System has batches but this node is inactive
        anomaly_score +=1

    label_val = 1 if anomaly_score >= 2 else 0
    label = np.array([label_val], dtype=np.int32)
    if label_val == 1:
        print(f"  Node {node_address_checksum} labeled ANOMALOUS for Batch Monitoring (score: {anomaly_score}). F:[{features[0]:.0f},{features[1]:.2f},{features[2]:.0f},{features[3]:.2f},{features[4]:.1f},{features[5]:.2f}], Sybil: {is_known_sybil}")
    return features, label

def generate_features_and_label_for_batch_monitoring(node_address_checksum, all_events, sybil_attack_log):
    features = np.zeros(NUM_BATCH_MONITORING_FEATURES, dtype=np.float32)
    anomaly_score = 0
    is_known_sybil = False

    # Check if the node is a known Sybil from the log
    if sybil_attack_log and "sybilNodes" in sybil_attack_log:
        for sybil_node_entry in sybil_attack_log["sybilNodes"]:
            if sybil_node_entry.get("address") == node_address_checksum:
                is_known_sybil = True
                break

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
        if num_proposed_failed > PROPOSER_MAX_FAILED_BATCHES_ABSOLUTE:
            anomaly_score +=1

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
        
        # Rough participation rate check (could be improved if we know all batches a validator was *selected* for)
        total_batches_in_system = len(all_events.get("BatchProposed", []))
        if total_batches_in_system > 0 and (num_batches_validated_participation / total_batches_in_system) < VALIDATOR_LOW_PARTICIPATION_THRESHOLD:
            if num_batches_proposed == 0: # Only penalize if not also a proposer (proposers might not validate much)
                 anomaly_score += 1

    # Check for Scenario B (Collusive Voting) and Scenario C (Counterfeit Batch) from sybil_attack_log
    if is_known_sybil and sybil_attack_log and "attackDetails" in sybil_attack_log:
        for attack_entry in sybil_attack_log["attackDetails"]:
            # Scenario B: Collusive Voting
            if attack_entry.get("scenario") == "B" and attack_entry.get("action") == "Sybil nodes vote for batch" \
               and node_address_checksum in attack_entry.get("sybilsInvolved", []):
                batch_id_attack = attack_entry.get("batchId")
                # Check if this node actually voted on this batch according to blockchain events
                corresponding_vote_event = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id_attack), None)
                if corresponding_vote_event:
                    num_collusive_votes += 1
                    anomaly_score += 2 # Strong indicator
                    print(f"  Node {node_address_checksum} detected in collusive voting (Scenario B) for batch {batch_id_attack}")
            
            # Scenario C: Proposing/Approving Counterfeit Batch
            if attack_entry.get("scenario") == "C":
                if attack_entry.get("action") == "Sybil proposes counterfeit batch" and attack_entry.get("proposer") == node_address_checksum:
                    anomaly_score += 3 # Very strong indicator
                    print(f"  Node {node_address_checksum} detected proposing counterfeit batch (Scenario C) for batch {attack_entry.get('batchId')}")
                elif attack_entry.get("action") == "Sybils approve counterfeit batch" and node_address_checksum in attack_entry.get("sybilsInvolved", []):
                    batch_id_attack_c = attack_entry.get("batchId")
                    corresponding_vote_event_c = next((ve for ve in validated_by_node if ve["args"]["batchId"] == batch_id_attack_c), None)
                    if corresponding_vote_event_c and corresponding_vote_event_c["args"]["approve"]:
                        anomaly_score += 2 # Strong indicator for approving
                        print(f"  Node {node_address_checksum} detected approving counterfeit batch (Scenario C) for batch {batch_id_attack_c}")

    # Update features (placeholder for new features, e.g., collusive vote count)
    # features[X] = float(num_collusive_votes) # Example, if a new feature slot is added
    
    # If node is neither proposer nor validator in any significant way
    if num_batches_proposed == 0 and num_batches_validated_participation == 0 and len(all_events.get("BatchProposed",[])) > 5 : # System has batches but this node is inactive
        anomaly_score +=1

    label_val = 1 if anomaly_score >= 2 else 0
    label = np.array([label_val], dtype=np.int32)
    if label_val == 1:
        print(f"  Node {node_address_checksum} labeled ANOMALOUS for Batch Monitoring (score: {anomaly_score}). F:[{features[0]:.0f},{features[1]:.2f},{features[2]:.0f},{features[3]:.2f},{features[4]:.1f},{features[5]:.2f}], Sybil: {is_known_sybil}")
    return features, label

def load_real_data_for_fl_client_batch_monitoring(client_id: str, target_node_addresses: list[str], bc_connector, sybil_attack_log):
    print(f"FL Client {client_id}: Loading real batch monitoring data for nodes: {target_node_addresses}")
    if not EVENT_CACHE["processed"]:
        fetch_and_cache_batch_events(bc_connector)
    if not EVENT_CACHE["processed"] or not any(EVENT_CACHE[k] for k in ["BatchProposed", "BatchValidated", "BatchCommitted"]):
        print(f"Warning: Event cache is empty or incomplete for client {client_id}. No data can be processed.")

    client_features_list = []
    client_labels_list = []

    for node_addr_orig in target_node_addresses:
        try:
            node_addr_checksum = bc_connector.w3.to_checksum_address(node_addr_orig)
            # print(f"  Client {client_id}: Processing node {node_addr_checksum} for batch monitoring")
            features, label = generate_features_and_label_for_batch_monitoring(node_addr_checksum, EVENT_CACHE, sybil_attack_log)
            client_features_list.append(features)
            client_labels_list.append(label)
            time.sleep(0.05) 
        except Exception as e:
            print(f"Error processing node {node_addr_orig} for client {client_id} batch_monitoring: {e}")

    if not client_features_list:
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    return dataset.batch(32)

def make_federated_data_batch_monitoring_real(all_target_node_addresses: list[str], num_fl_clients: int, sybil_attack_log):
    bc_connector = BlockchainConnector(rpc_url_override=RPC_URL_OVERRIDE)
    if "SupplyChainNFT" not in bc_connector.contracts or "BatchProcessing" not in bc_connector.contracts:
        print("Error: SupplyChainNFT or BatchProcessing contract not loaded...")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    fetch_and_cache_batch_events(bc_connector)
    if not EVENT_CACHE["processed"] or not any(EVENT_CACHE[k] for k in ["BatchProposed", "BatchValidated"]):
         print("Critical Error: Failed to fetch or cache batch events...")
         empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
         return [empty_ds for _ in range(num_fl_clients)]

    if not all_target_node_addresses:
        print("Warning: No target node addresses provided for batch monitoring...")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    nodes_per_client = len(all_target_node_addresses) // num_fl_clients
    extra_nodes = len(all_target_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_batch_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(all_target_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(all_target_node_addresses))
            client_node_list = all_target_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list:
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_BATCH_MONITORING_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            client_ds = load_real_data_for_fl_client_batch_monitoring(client_id, client_node_list, bc_connector, sybil_attack_log)
        client_datasets.append(client_ds)
    
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Batch Monitoring with Heuristic Labeling...")
    example_node_addresses = [
        # Add actual node addresses involved in batch processing on Amoy testnet
        # "0xProposerAddress1", "0xValidatorAddress1", "0xValidatorAddress2"
    ]
    # Simulate loading sybil_attack_log.json for testing
    mock_sybil_log_path = "../../SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json" # Adjust path as needed
    mock_sybil_attack_log = {}
    try:
        # Corrected path for loading within the FL_Model directory structure for testing
        # This assumes the script is run from within FL_Model/tff_batch_monitoring/
        # And sybil_attack_log.json is in SupplyChain_dapp/scripts/lifecycle_demo/
        # So, path should be ../../../SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json
        # Or adjust based on actual execution directory of this test block.
        # For now, let's assume it's relative to the Federated Learning directory for simplicity in testing.
        # This path needs to be correct relative to where this script is *executed* from during the test.
        # If executed from `Federated Learning/FL_Model/tff_batch_monitoring/`:
        test_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "..", "..", "..", "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json")
        
        # A more robust way if the script is in e.g. Federated Learning/FL_Model/tff_batch_monitoring
        # and the log is in SupplyChain_dapp/scripts/lifecycle_demo
        # The base for both is likely the project root. Let's assume a structure like:
        # ProjectRoot/
        #   Federated Learning/FL_Model/tff_batch_monitoring/real_data_preparation_batch_monitoring.py
        #   SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json
        
        # Path from current file to ProjectRoot: ../../..
        # Path from ProjectRoot to log: SupplyChain_dapp/scripts/lifecycle_demo/sybil_attack_log.json
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_approx = os.path.join(current_script_dir, "..", "..", "..") # Up three levels
        sybil_log_json_path_test = os.path.join(project_root_approx, "SupplyChain_dapp", "scripts", "lifecycle_demo", "sybil_attack_log.json")

        if os.path.exists(sybil_log_json_path_test):
            with open(sybil_log_json_path_test, "r", encoding='utf-8') as f:
                mock_sybil_attack_log = json.load(f)
            print(f"Successfully loaded mock sybil_attack_log.json for testing from {sybil_log_json_path_test}")
            if "sybilNodes" in mock_sybil_attack_log:
                 example_node_addresses.extend([node["address"] for node in mock_sybil_attack_log["sybilNodes"]])
        else:
            print(f"Mock sybil_attack_log.json not found at {sybil_log_json_path_test}. Proceeding with placeholders.")
    except Exception as e:
        print(f"Error loading mock sybil_attack_log.json for testing: {e}")

    if not example_node_addresses:
        print("Warning: example_node_addresses is empty. Using placeholders for structural test.")
        example_node_addresses.extend(["0x0000000000000000000000000000000000000001", "0x0000000000000000000000000000000000000002"])

    print(f"Using example node addresses: {list(set(example_node_addresses))} (unique)") # Ensure unique addresses

    num_clients_test = 1 
    federated_train_data_test = make_federated_data_batch_monitoring_real(list(set(example_node_addresses)), num_fl_clients=num_clients_test, sybil_attack_log=mock_sybil_attack_log)
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets for batch monitoring.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        anomalous_count = 0
        normal_count = 0
        for features_batch, labels_batch in client_dataset:
            for j_idx in range(features_batch.shape[0]):
                num_elements += 1
                print(f"  Client {i}, Sample {num_elements}: Features: {features_batch[j_idx].numpy()}, Label: {labels_batch[j_idx].numpy()}")
                if labels_batch[j_idx].numpy()[0] == 1:
                    anomalous_count +=1
                else:
                    normal_count +=1
        if num_elements == 0: print(f"  Client {i} has no data.")
        print(f"  Client {i} total samples: {num_elements}. Anomalous: {anomalous_count}, Normal: {normal_count}")

    print("\nReal data preparation test for Batch Monitoring with labeling complete.")

