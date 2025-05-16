# In data_preparation_p3_arbitrator.py
import tensorflow as tf
import numpy as np
import random

NUM_P3_ARB_FEATURES = 5
ELEMENT_SPEC_P3_ARB = (
    tf.TensorSpec(shape=(NUM_P3_ARB_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)
)

SIM_ARBITRATORS = [f"arb_{i}" for i in range(20)]
ARBITRATOR_PROFILES = {}
for i, arb_id in enumerate(SIM_ARBITRATORS):
    ARBITRATOR_PROFILES[arb_id] = {
        "is_biased": True if i % 4 == 0 else False,  # ~25% are biased
        "bias_factor": random.uniform(0.6, 0.9) if (i % 4 == 0) else random.uniform(0.4, 0.6)
    }

def simulate_arbitrator_performance(arbitrator_id, num_disputes_arbitrated=50):
    profile = ARBITRATOR_PROFILES[arbitrator_id]
    label = [1] if profile["is_biased"] else [0]
    
    # Simulate features based on bias
    # Feature 0: vote_alignment_with_outcome (biased might be lower if they vote against fair outcomes)
    f0 = random.uniform(0.3, 0.7) if profile["is_biased"] else random.uniform(0.6, 0.95)
    
    # Feature 1: vote_alignment_with_peers (biased might deviate more)
    f1 = random.uniform(0.4, 0.7) if profile["is_biased"] else random.uniform(0.7, 0.9)
    
    # Feature 2: disputes_participated
    f2 = float(num_disputes_arbitrated)
    
    # Feature 3 & 4: Simplified bias towards favoring one party in value (e.g. party A vs B)
    # Assume higher value for party A if biased towards A
    favored_A_value = random.uniform(1000, 5000) * profile["bias_factor"]
    favored_B_value = random.uniform(1000, 5000) * (1.0 - profile["bias_factor"])
    f3 = favored_A_value
    f4 = favored_B_value
    
    features = np.array([f0, f1, f2, f3, f4], dtype=np.float32)
    return features, np.array(label, dtype=np.int32)

def load_local_data_for_p3_arbitrator_client(client_id: str, assigned_arbitrators: list[str]):
    client_features = []
    client_labels = []
    
    for arb_id in assigned_arbitrators:
        features, label = simulate_arbitrator_performance(
            arb_id,
            num_disputes_arbitrated=random.randint(20, 100)
        )
        client_features.append(features)
        client_labels.append(label)
    
    if not client_features:  # Handle empty case
        return tf.data.Dataset.from_tensor_slices((
            np.zeros((0, NUM_P3_ARB_FEATURES), dtype=np.float32),
            np.zeros((0, 1), dtype=np.int32)
        ))
    
    # Ensure features are 2D (num_samples, num_features)
    features_array = np.array(client_features, dtype=np.float32)  # Shape: (num_samples, num_features)
    labels_array = np.array(client_labels, dtype=np.int32)  # Shape: (num_samples, 1)
    
    # Normalize features
    # Features 0-1: already in [0,1] range
    # Feature 2: normalize by max value (100)
    features_array[:, 2] = features_array[:, 2] / 100.0
    # Features 3-4: normalize by max value (5000)
    features_array[:, 3:] = features_array[:, 3:] / 5000.0
    
    # Debug prints
    print(f"\nClient {client_id} data stats:")
    print(f"Number of samples: {len(features_array)}")
    print(f"Features shape: {features_array.shape}")
    print(f"Labels shape: {labels_array.shape}")
    print(f"Features mean: {np.mean(features_array, axis=0)}")
    print(f"Features std: {np.std(features_array, axis=0)}")
    print(f"Label distribution: {np.bincount(labels_array.flatten())}")
    
    # Create dataset with proper shapes
    dataset = tf.data.Dataset.from_tensor_slices((
        features_array,  # Shape: (num_samples, num_features)
        labels_array     # Shape: (num_samples, 1)
    ))
    
    # Only shuffle, no batching here
    dataset = dataset.shuffle(1000)
    
    return dataset

def make_federated_data_p3_arbitrator(num_fl_clients=3):
    # Distribute arbitrators among FL clients
    random.shuffle(SIM_ARBITRATORS)
    arbitrators_per_fl_client = len(SIM_ARBITRATORS) // num_fl_clients
    client_datasets = []
    
    for i in range(num_fl_clients):
        start_idx = i * arbitrators_per_fl_client
        end_idx = (i + 1) * arbitrators_per_fl_client if i < num_fl_clients - 1 else len(SIM_ARBITRATORS)
        client_datasets.append(
            load_local_data_for_p3_arbitrator_client(
                f"fl_arb_client_{i}",
                SIM_ARBITRATORS[start_idx:end_idx]
            )
        )
    
    return client_datasets

# Add __main__ for testing this script independently


if __name__ == '__main__':
    print("Testing data_preparation_p3_arbitrator.py independently...")
    num_clients_to_test = 2
    federated_data = make_federated_data_p3_arbitrator(num_fl_clients=num_clients_to_test)
    print(f"Generated federated data for {len(federated_data)} clients.")
    for i, client_data in enumerate(federated_data):
        print(f"Client {i} data spec: {client_data.element_spec}")
        num_samples = 0
        for _ in client_data:
            num_samples += 1
        print(f"Client {i} has {num_samples} samples.")
        if num_samples > 0:
            for features, label in client_data.take(1):
                print(f"  First sample features shape: {features.shape}")
                print(f"  First sample label shape: {label.shape}")
                print(f"  First sample features: {features.numpy()}")
                print(f"  First sample label: {label.numpy()}")
                break
    print("Test complete.")

