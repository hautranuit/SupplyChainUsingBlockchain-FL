import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import numpy as np

# Corrected import for data preparation for this specific task
from data_preparation_p3_timeseries import make_federated_data_p3_timeseries, ELEMENT_SPEC_P3_TS, TIMESTEPS, NUM_P3_TS_FEATURES

# Corrected import for federated training and model definition for this specific task
from federated_training_p3_task import (
    build_weighted_fed_avg as build_weighted_fed_avg_p3, 
    preprocess_client_dataset, 
    detect_anomalies
)
from model_definition_p3_task import model_fn, create_keras_model 

nest_asyncio.apply()

def run_simulation_p3_timeseries(num_rounds=10, num_clients=3, batch_size=32):
    """Runs the federated learning simulation for Time-Series Anomaly Detection."""
    print("Starting Federated Learning Simulation for Time-Series Anomaly Detection...")
    
    print(f"Preparing data for {num_clients} Time-Series Anomaly FL clients...")
    train_data = make_federated_data_p3_timeseries(num_fl_clients=num_clients)
    print("Client datasets created.")

    print(f"Preprocessing client datasets with batch size {batch_size}...")
    train_data = [preprocess_client_dataset(ds, batch_size=batch_size) for ds in train_data]
    print("Client datasets preprocessed.")
    
    print("Building the federated training process (FedAvg) for Time-Series Anomaly Detection...")
    fed_avg = build_weighted_fed_avg_p3()
    print("Federated training process built.")

    print("Initializing the iterative process...")
    state = fed_avg.initialize()
    print("Initialization complete.")
    
    print(f"Starting {num_rounds} rounds of federated training...")
    for round_num in range(num_rounds):
        state, metrics = fed_avg.next(state, train_data)
        
        client_work_metrics = metrics.get("client_work", {})
        train_metrics = client_work_metrics.get("train", {})
        reconstruction_loss = train_metrics.get(
            "mean_squared_error", 
            train_metrics.get("loss", float("nan"))
        )

        print(f"Round {round_num + 1:2d}: "
              f"Reconstruction Loss={reconstruction_loss:.4f}")
    
    print("Federated training for Time-Series Anomaly Detection completed.")
    return state

def evaluate_timeseries_anomaly_detection(server_state, num_test_clients=1, batch_size=32, anomaly_threshold=0.1):
    """Evaluates the trained time-series autoencoder model for anomaly detection."""
    print("\nEvaluating Time-Series Anomaly Detection Model...")

    keras_model = create_keras_model()
    
    # Correctly extract weights from server_state.global_model_weights
    # The global_model_weights is an OrderedDict, typically with 'trainable' and 'non_trainable' keys.
    # We need to concatenate them in the order the Keras model expects.
    # For a simple sequential model with only trainable weights, this might be simpler.
    # Let's assume the weights are directly applicable or need to be structured.
    # TFF model_weights are usually a structure of (trainable, non_trainable) weights.
    # Keras model.set_weights expects a flat list of numpy arrays.
    
    # Get the TFF model's weights structure
    tff_model_weights = fed_avg.get_model_weights(server_state) # Use the iterative process to get weights structure
    
    # Apply the weights to the Keras model
    tff_model_weights.assign_weights_to(keras_model)

    print(f"Generating test data for {num_test_clients} client(s)...")
    raw_test_data = make_federated_data_p3_timeseries(num_fl_clients=num_test_clients)
    
    all_test_input_sequences = []

    for client_dataset in raw_test_data:
        for input_seq, target_seq in client_dataset:
            all_test_input_sequences.append(input_seq.numpy())
    
    if not all_test_input_sequences:
        print("No test data generated for evaluation.")
        return

    all_test_input_sequences_np = np.array(all_test_input_sequences)

    print(f"Total test sequences: {all_test_input_sequences_np.shape[0]}")

    print(f"Detecting anomalies with threshold: {anomaly_threshold}...")
    anomalous_flags = detect_anomalies(keras_model, all_test_input_sequences_np, threshold=anomaly_threshold)
    num_anomalies_detected = np.sum(anomalous_flags.numpy())
    total_samples = anomalous_flags.shape[0]
    anomaly_rate = num_anomalies_detected / total_samples if total_samples > 0 else 0

    print(f"  Total test samples: {total_samples}")
    print(f"  Number of anomalies detected: {num_anomalies_detected}")
    print(f"  Anomaly rate on test data: {anomaly_rate:.2%}")
    print("Note: True anomaly labels are not explicitly tracked in this basic test data generation for evaluation simplicity.")
    print("Evaluation of anomaly detection finished.")

# Global reference to fed_avg for get_model_weights
fed_avg = None

if __name__ == '__main__':
    print("Running Time-Series Anomaly Detection Simulation (run_simulation_p3_task.py in node_behavior_timeseries)...")
    
    # Build fed_avg globally so it can be used in evaluation
    fed_avg = build_weighted_fed_avg_p3()

    final_server_state = run_simulation_p3_timeseries(num_rounds=5, num_clients=2, batch_size=16)
    
    evaluate_timeseries_anomaly_detection(final_server_state, num_test_clients=1, batch_size=16, anomaly_threshold=0.05)
    print("\nTime-Series Anomaly Detection Simulation and Evaluation finished.")

