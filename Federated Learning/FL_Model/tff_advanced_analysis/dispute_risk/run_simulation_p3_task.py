import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import collections # For OrderedDict if used in metrics
import numpy as np

# Corrected import for data preparation for this specific task
from data_preparation_p3_dispute import make_federated_data_p3_dispute, ELEMENT_SPEC_P3_DIS
# Corrected import for federated training and model definition for this specific task
from federated_training_p3_task import build_weighted_fed_avg as build_weighted_fed_avg_p3, preprocess_client_dataset
from model_definition_p3_task import model_fn # model_fn is already specific to the task

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily.
nest_asyncio.apply()

def run_simulation_p3_dispute(num_rounds=10, num_clients=3, disputes_per_client=100, batch_size=32):
    """Runs the federated learning simulation for High-Risk Dispute Prediction."""
    print("Starting Federated Learning Simulation for High-Risk Dispute Prediction...")
    
    print(f"Preparing data for {num_clients} High-Risk Dispute FL clients, with {disputes_per_client} disputes each...")
    train_data_orig = make_federated_data_p3_dispute(num_fl_clients=num_clients, disputes_per_client=disputes_per_client)
    print("Client datasets created.")

    print(f"Preprocessing client datasets with batch size {batch_size}...")
    processed_train_data = [preprocess_client_dataset(ds, batch_size=batch_size) for ds in train_data_orig]
    print("Client datasets preprocessed.")

    if processed_train_data:
        print(f"DEBUG: Processed train_data[0] element_spec: {processed_train_data[0].element_spec}")
        for batch_idx, (batch_features, batch_labels) in enumerate(processed_train_data[0].take(1)):
            print(f"  DEBUG: Client 0, Batch {batch_idx} - Features shape: {tf.shape(batch_features).numpy()}, Labels shape: {tf.shape(batch_labels).numpy()}")
            if tf.size(batch_features) > 0:
                print(f"  DEBUG: Client 0, Batch {batch_idx} - First feature vector in batch: {batch_features.numpy()[0]}")
            else:
                print(f"  DEBUG: Client 0, Batch {batch_idx} - Empty batch features.")
            break
    
    print("Building the federated training process (FedAvg) for High-Risk Dispute Prediction...")
    fed_avg = build_weighted_fed_avg_p3()
    print("Federated training process built.")

    print("Initializing the iterative process...")
    state = fed_avg.initialize()
    print("Initialization complete.")
    
    print(f"Starting {num_rounds} rounds of federated training...")
    for round_num in range(num_rounds):
        state, metrics = fed_avg.next(state, processed_train_data)
        
        client_work_metrics = metrics.get('client_work', {})
        train_metrics = client_work_metrics.get('train', {})
        
        accuracy = train_metrics.get('binary_accuracy', train_metrics.get('accuracy', float('nan')))
        auc = train_metrics.get('auc', float('nan'))
        loss = train_metrics.get('loss', float('nan'))

        print(f"Round {round_num + 1:2d}: "
              f"Accuracy={accuracy:.4f}, "
              f"AUC={auc:.4f}, "
              f"Loss={loss:.4f}")
    
    print("Federated training for High-Risk Dispute Prediction completed.")
    return state

if __name__ == '__main__':
    print("Running High-Risk Dispute Simulation (run_simulation_p3_task.py in dispute_risk)...")
    final_state_dispute_risk = run_simulation_p3_dispute(num_rounds=5, num_clients=2, disputes_per_client=50, batch_size=16)
    print("\nHigh-Risk Dispute Simulation finished.")

