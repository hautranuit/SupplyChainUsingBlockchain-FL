import tensorflow as tf
import numpy as np
import pandas as pd  # For potential CSV handling later
import random

# Define the number of features for Phase 2. This needs to match your feature engineering.
NUM_PHASE2_FEATURES = 6
ELEMENT_SPEC_PHASE2 = (
    tf.TensorSpec(shape=(NUM_PHASE2_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Label: 0 for normal, 1 for anomalous/collusive
)

# --- Simulation of Batch Processing Environment ---
NUM_SIM_VALIDATORS = 10
NUM_SIM_PROPOSERS = 5
SIM_VALIDATOR_IDS = [f"val_{i}" for i in range(NUM_SIM_VALIDATORS)]
SIM_PROPOSER_IDS = [f"prop_{i}" for i in range(NUM_SIM_PROPOSERS)]

BEHAVIOR_PROFILES = {
    "normal_validator": {"collusion_tendency": 0.1, "error_rate": 0.05, "label": 0},
    "collusive_validator": {"collusion_tendency": 0.9, "error_rate": 0.05, "label": 1},
    "faulty_validator": {"collusion_tendency": 0.1, "error_rate": 0.5, "label": 1},
    "normal_proposer": {"success_bias": 0.8, "low_quality_rate": 0.1, "label": 0},
    "bad_proposer": {"success_bias": 0.2, "low_quality_rate": 0.7, "label": 1}
}

NODE_PROFILES = {}
for i, vid in enumerate(SIM_VALIDATOR_IDS):
    if i < NUM_SIM_VALIDATORS // 2:
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["normal_validator"]
    elif i < NUM_SIM_VALIDATORS * 0.8:
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["collusive_validator"]
    else:
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["faulty_validator"]

for i, pid in enumerate(SIM_PROPOSER_IDS):
    if i < NUM_SIM_PROPOSERS // 2:
        NODE_PROFILES[pid] = BEHAVIOR_PROFILES["normal_proposer"]
    else:
        NODE_PROFILES[pid] = BEHAVIOR_PROFILES["bad_proposer"]

COLLUSION_GROUP_A = [SIM_VALIDATOR_IDS[i] for i in 
                     range(NUM_SIM_VALIDATORS // 2, int(NUM_SIM_VALIDATORS * 0.8))]

def simulate_batch_events(num_batches=100):
    batch_log = []
    for batch_id in range(num_batches):
        proposer = random.choice(SIM_PROPOSER_IDS)
        is_low_quality_batch = random.random() < NODE_PROFILES[proposer]["low_quality_rate"]
        num_selected_validators = min(NUM_SIM_VALIDATORS, max(3, NUM_SIM_VALIDATORS // 2))
        selected_validators = random.sample(SIM_VALIDATOR_IDS, num_selected_validators)
        votes = {}
        num_approvals = 0
        for validator_id in selected_validators:
            profile = NODE_PROFILES[validator_id]
            vote_approve = True
            if validator_id in COLLUSION_GROUP_A and proposer == SIM_PROPOSER_IDS[-1]:
                vote_approve = random.random() < profile["collusion_tendency"]
            elif is_low_quality_batch:
                vote_approve = random.random() < (profile["collusion_tendency"] * 0.5 + profile["error_rate"])
            else:
                vote_approve = random.random() > profile["error_rate"]
            votes[validator_id] = vote_approve
            if vote_approve:
                num_approvals += 1
        batch_outcome_committed = num_approvals > num_selected_validators / 2
        batch_log.append({
            "batch_id": batch_id,
            "proposer": proposer,
            "is_low_quality": is_low_quality_batch,
            "selected_validators": selected_validators,
            "votes": votes,
            "committed": batch_outcome_committed
        })
    return batch_log

def extract_features_from_log(batch_log, target_node_id):
    node_batches = [b for b in batch_log if target_node_id == b["proposer"] or 
                   target_node_id in b["selected_validators"]]
    if not node_batches:
        return None
    label = NODE_PROFILES[target_node_id]["label"]
    features = np.zeros(NUM_PHASE2_FEATURES, dtype=np.float32)
    if target_node_id.startswith("val_"):
        participated = 0
        consistent_votes = 0
        agreed_majority = 0
        solo_dissents = 0
        for batch in node_batches:
            if target_node_id not in batch["selected_validators"]:
                continue
            participated += 1
            my_vote = batch["votes"][target_node_id]
            if my_vote == batch["committed"]:
                consistent_votes += 1
            num_approvals = sum(batch["votes"].values())
            majority_vote_approves = num_approvals > len(batch["selected_validators"]) / 2
            if my_vote == majority_vote_approves:
                agreed_majority += 1
            if (len(batch["selected_validators"]) > 1 and 
                my_vote != majority_vote_approves and
                ((my_vote and num_approvals == 1) or 
                 (not my_vote and num_approvals == len(batch["selected_validators"])-1))):
                solo_dissents += 1
        features[0] = consistent_votes / participated if participated else 0
        features[1] = agreed_majority / participated if participated else 0
        features[2] = solo_dissents / participated if participated else 0
        features[3] = participated / len(batch_log)
    elif target_node_id.startswith("prop_"):
        proposed = 0
        succeeded = 0
        flagged = 0
        for batch in node_batches:
            if target_node_id != batch["proposer"]:
                continue
            proposed += 1
            if batch["committed"]:
                succeeded += 1
            else:
                flagged += 1
        features[0] = succeeded / proposed if proposed else 0
        features[1] = flagged / proposed if proposed else 0
        features[2] = proposed / len(batch_log)
    return features, np.array([label], dtype=np.int32)

GLOBAL_BATCH_LOG = simulate_batch_events(num_batches=500)

def load_local_data_for_phase2_client(client_id: str, node_ids_for_client: list[str]):
    print(f"Client {client_id}: Extracting features for nodes: {node_ids_for_client}")
    client_features = []
    client_labels = []
    for node_id in node_ids_for_client:
        result = extract_features_from_log(GLOBAL_BATCH_LOG, node_id)
        if result:
            features, label = result
            client_features.append(features)
            client_labels.append(label)
    if not client_features:
        print(f"Warning: Client {client_id} had no data. Creating dummy data.")
        current_features_array = np.zeros((1, NUM_PHASE2_FEATURES), dtype=np.float32)
        current_labels_array = np.array([[0]], dtype=np.int32)
    else:
        current_features_array = np.array(client_features, dtype=np.float32)
        current_labels_array = np.array(client_labels, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((current_features_array, current_labels_array))
    # Apply shuffle and batch to all datasets, including dummy ones
    dataset = dataset.shuffle(1000).batch(32)
    return dataset

def make_federated_data_phase2(num_fl_clients=3):
    all_sim_nodes = SIM_VALIDATOR_IDS + SIM_PROPOSER_IDS
    random.shuffle(all_sim_nodes)
    nodes_per_fl_client = len(all_sim_nodes) // num_fl_clients
    client_datasets = []
    for i in range(num_fl_clients):
        start_idx = i * nodes_per_fl_client
        end_idx = (i + 1) * nodes_per_fl_client if i < num_fl_clients - 1 else len(all_sim_nodes)
        fl_client_id = f"fl_client_{i}"
        assigned_nodes = all_sim_nodes[start_idx:end_idx]
        if not assigned_nodes:
            print(f"Warning: FL Client {fl_client_id} assigned no nodes. Skipping.")
            continue
        client_datasets.append(load_local_data_for_phase2_client(fl_client_id, assigned_nodes))
    return client_datasets

if __name__ == '__main__':
    print("Testing Phase 2 Data Preparation...")
    federated_data = make_federated_data_phase2(num_fl_clients=3)
    print(f"\nCreated {len(federated_data)} FL client datasets for Phase 2.")
    for i, ds in enumerate(federated_data):
        print(f"FL Client {i} dataset element spec: {ds.element_spec}")
        # Check the batched spec
        expected_feature_shape = [None, NUM_PHASE2_FEATURES]
        expected_label_shape = [None, 1]
        actual_feature_shape = ds.element_spec[0].shape.as_list()
        actual_label_shape = ds.element_spec[1].shape.as_list()
        # Allow for the batch dimension to be fixed if dataset size is less than batch size
        if actual_feature_shape[0] is not None and actual_feature_shape[0] < 32:
             pass # This is fine for small datasets
        else:
            assert actual_feature_shape[0] is None or actual_feature_shape[0] == 32, \
                f"Feature spec batch dimension mismatch for client {i}: expected None or 32, got {actual_feature_shape[0]}"
        assert actual_feature_shape[1:] == expected_feature_shape[1:], \
            f"Feature spec mismatch for client {i}: expected {expected_feature_shape[1:]}, got {actual_feature_shape[1:]}"
        if actual_label_shape[0] is not None and actual_label_shape[0] < 32:
            pass
        else:
            assert actual_label_shape[0] is None or actual_label_shape[0] == 32, \
                f"Label spec batch dimension mismatch for client {i}: expected None or 32, got {actual_label_shape[0]}"
        assert actual_label_shape[1:] == expected_label_shape[1:], \
            f"Label spec mismatch for client {i}: expected {expected_label_shape[1:]}, got {actual_label_shape[1:]}"
        num_batches_in_ds = 0
        for features_batch, labels_batch in ds:
            num_batches_in_ds +=1
            # print(f" Features batch shape: {features_batch.shape}, Labels batch shape: {labels_batch.shape}")
        print(f" Client {i} has {num_batches_in_ds} batches.")
    print("Phase 2 Data Preparation Test Complete.")

