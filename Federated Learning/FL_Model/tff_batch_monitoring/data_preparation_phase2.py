import tensorflow as tf
import numpy as np
import pandas as pd  # For potential CSV handling later
import random

# Define the number of features for Phase 2. This needs to match your feature engineering.
# Let's assume 6 features for validators and 5 for proposers for this example.
# For simplicity in a unified model, we might pad or select a common number, e.g., 6.
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

# Define some behavioral profiles for simulation
BEHAVIOR_PROFILES = {
    "normal_validator": {"collusion_tendency": 0.1, "error_rate": 0.05, "label": 0},
    "collusive_validator": {"collusion_tendency": 0.9, "error_rate": 0.05, "label": 1},
    "faulty_validator": {"collusion_tendency": 0.1, "error_rate": 0.5, "label": 1},
    "normal_proposer": {"success_bias": 0.8, "low_quality_rate": 0.1, "label": 0},
    "bad_proposer": {"success_bias": 0.2, "low_quality_rate": 0.7, "label": 1}
}

# Assign profiles to simulated nodes (can be more dynamic)
NODE_PROFILES = {}
for i, vid in enumerate(SIM_VALIDATOR_IDS):
    if i < NUM_SIM_VALIDATORS // 2:  # Half normal
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["normal_validator"]
    elif i < NUM_SIM_VALIDATORS * 0.8:  # Some collusive
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["collusive_validator"]
    else:  # Some faulty
        NODE_PROFILES[vid] = BEHAVIOR_PROFILES["faulty_validator"]

for i, pid in enumerate(SIM_PROPOSER_IDS):
    if i < NUM_SIM_PROPOSERS // 2:
        NODE_PROFILES[pid] = BEHAVIOR_PROFILES["normal_proposer"]
    else:
        NODE_PROFILES[pid] = BEHAVIOR_PROFILES["bad_proposer"]

COLLUSION_GROUP_A = [SIM_VALIDATOR_IDS[i] for i in 
                     range(NUM_SIM_VALIDATORS // 2, int(NUM_SIM_VALIDATORS * 0.8))]

def simulate_batch_events(num_batches=100):
    """Simulates a series of batch proposal and validation events."""
    batch_log = []  # Store (proposer, selected_validators, votes, outcome)
    for batch_id in range(num_batches):
        proposer = random.choice(SIM_PROPOSER_IDS)
        # Simulate proposer quality for this batch
        is_low_quality_batch = random.random() < NODE_PROFILES[proposer]["low_quality_rate"]
        num_selected_validators = min(NUM_SIM_VALIDATORS, max(3, NUM_SIM_VALIDATORS // 2))
        selected_validators = random.sample(SIM_VALIDATOR_IDS, num_selected_validators)
        votes = {}  # validator_id: vote (True for approve, False for reject)
        num_approvals = 0
        
        for validator_id in selected_validators:
            profile = NODE_PROFILES[validator_id]
            # Collusive behavior: if proposer is "bad_proposer" and validator is "collusive_validator"
            # or if validator is in a specific collusion group and proposer is part of their scheme
            vote_approve = True
            if validator_id in COLLUSION_GROUP_A and proposer == SIM_PROPOSER_IDS[-1]:  # Collude for last proposer
                vote_approve = random.random() < profile["collusion_tendency"]
            elif is_low_quality_batch:  # Tend to reject low quality unless colluding or faulty
                vote_approve = random.random() < (profile["collusion_tendency"] * 0.5 + profile["error_rate"])
            else:  # Tend to approve high quality unless faulty
                vote_approve = random.random() > profile["error_rate"]
            
            votes[validator_id] = vote_approve
            if vote_approve:
                num_approvals += 1
        
        # Simplified outcome: requires >50% approval (superMajority not fully modeled here)
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
    """Extracts features for a specific node from the batch log."""
    # This is a simplified feature extraction. Real one would be more complex.
    node_batches = [b for b in batch_log if target_node_id == b["proposer"] or 
                   target_node_id in b["selected_validators"]]
    if not node_batches:
        return None  # Node didn't participate
    
    label = NODE_PROFILES[target_node_id]["label"]
    features = np.zeros(NUM_PHASE2_FEATURES, dtype=np.float32)
    
    if target_node_id.startswith("val_"):  # Validator features
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
        features[3] = participated / len(batch_log)  # Participation rate
        # features 4, 5 can be other validator metrics like approval for low_rep proposers
    
    elif target_node_id.startswith("prop_"):  # Proposer features
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
        features[2] = proposed / len(batch_log)  # Proposal rate
        # features 3, 4 can be other proposer metrics
    
    return features, np.array([label], dtype=np.int32)

GLOBAL_BATCH_LOG = simulate_batch_events(num_batches=500)  # Generate a global log once

def load_local_data_for_phase2_client(client_id: str, node_ids_for_client: list[str]):
    """Generates client dataset by extracting features for its assigned nodes from the global log."""
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
        dummy_features = np.zeros((1, NUM_PHASE2_FEATURES), dtype=np.float32)
        dummy_labels = np.array([[0]], dtype=np.int32)
        return tf.data.Dataset.from_tensor_slices((dummy_features, dummy_labels))
    
    # Convert to numpy arrays and ensure correct shapes
    features_array = np.array(client_features, dtype=np.float32)
    labels_array = np.array(client_labels, dtype=np.int32)
    
    # Create dataset with correct shapes
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    
    # Add batching and shuffling
    dataset = dataset.shuffle(1000).batch(32)
    
    return dataset

def make_federated_data_phase2(num_fl_clients=3):
    """Creates federated datasets. Each FL client gets a subset of all simulated nodes."""
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
        num_elements = 0
        for features, label in ds:
            num_elements += 1
            # print(f" Features: {features.numpy()}, Label: {label.numpy()}")
        print(f" Client {i} has {num_elements} data points (nodes).")
        
        if num_elements > 0:
            assert ds.element_spec[0].shape.as_list() == [NUM_PHASE2_FEATURES], \
                f"Feature spec mismatch for client {i}"
            assert ds.element_spec[1].shape.as_list() == [1], \
                f"Label spec mismatch for client {i}"
    
    print("Phase 2 Data Preparation Test Complete.")