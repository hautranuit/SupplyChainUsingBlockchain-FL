# In data_preparation_p3_dispute.py
import tensorflow as tf
import numpy as np
import random

NUM_P3_DR_FEATURES = 6
NUM_P3_DIS_FEATURES = NUM_P3_DR_FEATURES
ELEMENT_SPEC_P3_DR = (
    tf.TensorSpec(shape=(NUM_P3_DR_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=np.int32)
)
ELEMENT_SPEC_P3_DIS = ELEMENT_SPEC_P3_DR

def simulate_dispute_characteristics(dispute_id):
    is_high_risk = random.random() < 0.3  # ~30% are high-risk
    label = [1] if is_high_risk else [0]
    
    # Simulate features
    f0_value = random.uniform(100, 100000) * (1.5 if is_high_risk else 1.0)  # Higher value if high-risk
    f1_parties = random.randint(2, 5)
    f2_avg_rep = random.uniform(10, 100)
    f3_rep_diff = random.uniform(0, 50) * (1.2 if is_high_risk else 0.8)
    f4_evidence = random.randint(1, 20) * (0.7 if is_high_risk else 1.3)  # Less evidence if high-risk (e.g. fraud)
    f5_prior_risk_initiator = random.uniform(0, 1) * (1.8 if is_high_risk else 0.5)
    
    features = np.array([
        f0_value, f1_parties, f2_avg_rep, f3_rep_diff, f4_evidence,
        f5_prior_risk_initiator
    ], dtype=np.float32)
    
    return features, np.array(label, dtype=np.int32)

def load_local_data_for_p3_dispute_client(
    client_id: str,
    num_disputes_to_generate=100
):
    # Each client might observe/report a set of disputes
    client_features = []
    client_labels = []
    
    for i in range(num_disputes_to_generate):
        features, label = simulate_dispute_characteristics(f"dispute_{client_id}_{i}")
        client_features.append(features)
        client_labels.append(label)
    
    return tf.data.Dataset.from_tensor_slices((
        np.array(client_features),
        np.array(client_labels)
    ))

def make_federated_data_p3_dispute(num_fl_clients=3, disputes_per_client=100):
    return [
        load_local_data_for_p3_dispute_client(
            f"fl_disp_client_{i}",
            disputes_per_client
        ) for i in range(num_fl_clients)
    ]

# Add __main__ for testing this script independently    


if __name__ == '__main__':
    print("Testing data_preparation_p3_dispute.py independently...")
    num_clients_to_test = 2
    disputes_per_client_test = 50
    federated_data = make_federated_data_p3_dispute(num_fl_clients=num_clients_to_test, disputes_per_client=disputes_per_client_test)
    print(f"Generated federated data for {len(federated_data)} clients.")
    for i, client_data in enumerate(federated_data):
        print(f"Client {i} data spec: {client_data.element_spec}")
        num_samples = 0
        # Iterate through the dataset to count samples
        for _ in client_data:
            num_samples += 1
        print(f"Client {i} has {num_samples} samples (expected {disputes_per_client_test}).")
        if num_samples > 0:
            for features, label in client_data.take(1):
                print(f"  First sample features: {features.numpy()}")
                print(f"  First sample label: {label.numpy()}")
                break
    print("Test complete.")

