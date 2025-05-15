import tensorflow as tf
import numpy as np
# from sklearn.preprocessing import StandardScaler # Optional: if you use it

# Define an OrderDict for feature specification for TFF
# This describes the structure of a single data point (features, label)
# Features: (num_features,), Label: (1,) for binary classification
ELEMENT_SPEC = (
    tf.TensorSpec(shape=(5,), dtype=tf.float32),  # Features (5 in this example)
    tf.TensorSpec(shape=(1,), dtype=tf.int32)     # Labels (single label for binary classification)
)
NUM_FEATURES = 5  # Define this globally for consistency

def load_local_data_for_client(client_id: str, num_samples=100):
    """Simulates loading and preprocessing data for a single client.
    In a real scenario, this function would:
        1. Connect to the blockchain (e.g., using web3.py).
        2. Query NodeManagement.sol for node registration details (timestamp, role, type).
        3. Query transaction history for initial activity of nodes this client interacts with.
        4. Extract features: e.g., registration_age_days, transaction_frequency, diversity_of_interactions.
        5. Create labels: e.g., 0 for normal, 1 for suspicious (initially, this might be manually labeled or based on heuristics).
    """
    print(f"Client {client_id}: Simulating local data loading and preprocessing...")
    
    # Example: num_samples, NUM_FEATURES features
    X_local = np.random.rand(num_samples, NUM_FEATURES).astype(np.float32)
    # Labels should be shape (num_samples, 1) for binary crossentropy with from_logits=False
    y_local = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.int32)

    # Simulate some client-specific data variation (non-IID)
    if client_id == "client_1":
        print(f"Client {client_id}: Introducing data variation.")
        X_local[:num_samples//2, 0] += 0.7  # Make some features different for client 1
        y_local[:num_samples//2] = 1  # More suspicious samples for client 1
    elif client_id == "client_2":
        print(f"Client {client_id}: Introducing data variation.")
        X_local[num_samples//4:num_samples//2, 1] -= 0.5
        y_local[num_samples//4:num_samples//2] = 0

    # Optional: Feature Scaling (StandardScaler example)
    # scaler = StandardScaler()
    # X_local = scaler.fit_transform(X_local) # Fit scaler on training data only in real scenario

    # Create tf.data.Dataset from the client's data
    # TFF expects datasets to yield batches. Here, we make each client's full data a single batch for simplicity.
    # In practice, you would use .batch(BATCH_SIZE) on the dataset *before* federated processing.
    # The dataset should yield tuples matching ELEMENT_SPEC (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices((X_local, y_local))
    # For TFF, it's often better to batch within the TFF computation or prepare client datasets to be pre-batched.
    # For this example, we'll return the unbatched dataset and batch it later if needed by the TFF process.
    return dataset

def make_federated_data(client_ids: list[str], num_samples_per_client=100):
    """Creates a list of tf.data.Dataset objects for TFF simulation."""
    return [load_local_data_for_client(client_id, num_samples_per_client) for client_id in client_ids]

if __name__ == '__main__':
    # Test the data preparation
    print("Testing data preparation...")
    CLIENT_IDS_TEST = ["client_0", "client_1", "client_2"]
    federated_train_data_test = make_federated_data(CLIENT_IDS_TEST)
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {CLIENT_IDS_TEST[i]} dataset element spec: {client_dataset.element_spec}")
        # Take one element (which is a batch of all samples for this client in this setup)
        for features, labels in client_dataset.take(1):
            print(f" Features shape: {features.shape}, Labels shape: {labels.shape}")
            print(f" First feature vector: {features.numpy()[0]}")
            print(f" First label: {labels.numpy()[0]}")
        
        # Verify it matches ELEMENT_SPEC
        assert client_dataset.element_spec[0].shape.as_list() == [NUM_FEATURES], f"Feature spec mismatch for client {i}"
        assert client_dataset.element_spec[1].shape.as_list() == [1], f"Label spec mismatch for client {i}"
    
    print("\nData preparation test complete.")