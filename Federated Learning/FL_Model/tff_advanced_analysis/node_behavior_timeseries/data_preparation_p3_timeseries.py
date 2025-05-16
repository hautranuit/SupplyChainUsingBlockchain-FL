# In data_preparation_p3_timeseries.py
import tensorflow as tf
import numpy as np
import random

NUM_P3_TS_FEATURES = 3
TIMESTEPS = 30  # e.g., 30 days of data

# For an autoencoder, labels are the same as inputs, or use reconstruction error for anomaly scoring
ELEMENT_SPEC_P3_TS = (
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32),  # Input sequence
    tf.TensorSpec(shape=(TIMESTEPS, NUM_P3_TS_FEATURES), dtype=tf.float32)   # Output sequence (for autoencoder)
)

SIM_NODES_TS = [f"ts_node_{i}" for i in range(50)]

def generate_node_timeseries(node_id):
    # Normal behavior
    base_tx_freq = random.uniform(5, 20)
    base_tx_val = random.uniform(100, 500)
    base_new_interact = random.uniform(1, 5)
    sequence = []
    
    is_anomalous_node = random.random() < 0.2  # 20% of nodes exhibit anomaly at some point
    anomaly_start_step = (
        random.randint(TIMESTEPS // 2, TIMESTEPS - 5) 
        if is_anomalous_node 
        else TIMESTEPS
    )
    
    for step in range(TIMESTEPS):
        is_anomaly_now = is_anomalous_node and step >= anomaly_start_step
        factor = 3.0 if is_anomaly_now else 1.0  # Anomaly makes values spike
        noise = np.random.normal(0, 0.1, NUM_P3_TS_FEATURES)
        
        f0 = max(0, base_tx_freq * factor * (1 + noise[0]))
        f1 = max(0, base_tx_val * (factor if random.random() > 0.5 else 1/factor) * (1 + noise[1]))  # Value might spike or drop
        f2 = max(0, base_new_interact * factor * (1 + noise[2]))
        
        sequence.append([f0, f1, f2])
    
    seq_array = np.array(sequence, dtype=np.float32)
    return seq_array, seq_array  # Input and target are same for autoencoder

def load_local_data_for_p3_timeseries_client(
    client_id: str,
    assigned_nodes: list[str]
):
    client_input_seqs = []
    client_target_seqs = []
    
    for node_id in assigned_nodes:
        input_s, target_s = generate_node_timeseries(node_id)
        client_input_seqs.append(input_s)
        client_target_seqs.append(target_s)
    
    if not client_input_seqs:  # Handle empty case
        return tf.data.Dataset.from_tensor_slices((
            np.zeros((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32),
            np.zeros((0, TIMESTEPS, NUM_P3_TS_FEATURES), dtype=np.float32)
        ))
    
    return tf.data.Dataset.from_tensor_slices((
        np.array(client_input_seqs),
        np.array(client_target_seqs)
    ))

def make_federated_data_p3_timeseries(num_fl_clients=3):
    random.shuffle(SIM_NODES_TS)
    nodes_per_fl_client = len(SIM_NODES_TS) // num_fl_clients
    client_datasets = []
    
    for i in range(num_fl_clients):
        start_idx = i * nodes_per_fl_client
        end_idx = (
            (i + 1) * nodes_per_fl_client 
            if i < num_fl_clients - 1 
            else len(SIM_NODES_TS)
        )
        client_datasets.append(
            load_local_data_for_p3_timeseries_client(
                f"fl_ts_client_{i}",
                SIM_NODES_TS[start_idx:end_idx]
            )
        )
    
    return client_datasets

# Add __main__ for testing this script independently


if __name__ == '__main__':
    print("Testing data_preparation_p3_timeseries.py independently...")
    num_clients_to_test = 2
    federated_data = make_federated_data_p3_timeseries(num_fl_clients=num_clients_to_test)
    print(f"Generated federated data for {len(federated_data)} clients.")
    for i, client_data in enumerate(federated_data):
        print(f"Client {i} data spec: {client_data.element_spec}")
        num_samples = 0
        # Iterate through the dataset to count samples
        for _ in client_data:
            num_samples += 1
        print(f"Client {i} has {num_samples} samples.")
        if num_samples > 0:
            # For timeseries, data is (input_seq, target_seq)
            for input_seq, target_seq in client_data.take(1):
                print(f"  First sample input_seq shape: {input_seq.shape}, dtype: {input_seq.dtype}")
                print(f"  First sample target_seq shape: {target_seq.shape}, dtype: {target_seq.dtype}")
                break
    print("Test complete.")

