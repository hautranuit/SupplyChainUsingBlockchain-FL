import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import collections
import numpy as np

# Corrected import for data preparation for this specific task
from data_preparation_p3_arbitrator import make_federated_data_p3_arbitrator, ELEMENT_SPEC_P3_ARB
# Corrected import for federated training and model definition for this specific task
from federated_training_p3_task import build_weighted_fed_avg as build_weighted_fed_avg_p3, preprocess_client_dataset
from model_definition_p3_task import model_fn # model_fn is already specific to the task

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily.
nest_asyncio.apply()

def run_simulation_p3_arbitrator(num_rounds=10, num_clients=3, batch_size=32):
    """Runs the federated learning simulation for Arbitrator Bias detection."""
    print("Starting Federated Learning Simulation for Arbitrator Bias Detection...")
    
    print(f"Preparing data for {num_clients} Arbitrator Bias FL clients...")
    train_data = make_federated_data_p3_arbitrator(num_fl_clients=num_clients)
    print("Client datasets created.")

    # Optional: Keep some minimal debug for data shape if needed, but remove extensive prints for final version
    # print(f"Original train_data length: {len(train_data)}")
    # if train_data:
    #     print(f"Original train_data[0] element_spec: {train_data[0].element_spec}")

    print(f"Preprocessing client datasets with batch size {batch_size}...")
    train_data = [preprocess_client_dataset(ds, batch_size=batch_size) for ds in train_data]
    print("Client datasets preprocessed.")

    # Optional: Keep some minimal debug for preprocessed data if needed
    # print(f"Preprocessed train_data length: {len(train_data)}")
    # if train_data:
    #     print(f"Preprocessed train_data[0] element_spec: {train_data[0].element_spec}")
    
    print("Building the federated training process (FedAvg) for Arbitrator Bias...")
    fed_avg = build_weighted_fed_avg_p3(model_fn)
    print("Federated training process built.")

    print("Initializing the iterative process...")
    state = fed_avg.initialize()
    print("Initialization complete.")
    
    print(f"Starting {num_rounds} rounds of federated training...")
    for round_num in range(num_rounds):
        state, metrics = fed_avg.next(state, train_data)
        
        # print(f"Round {round_num + 1:2d} raw metrics: {metrics}") # Keep for one more run if needed, then remove

        # Corrected metric extraction from the nested OrderedDict
        client_work_metrics = metrics.get('client_work', {})
        train_metrics = client_work_metrics.get('train', {})
        
        accuracy = train_metrics.get('binary_accuracy', train_metrics.get('accuracy', float('nan')))
        auc = train_metrics.get('auc', float('nan'))
        loss = train_metrics.get('loss', float('nan'))

        print(f"Round {round_num + 1:2d}: "
              f"Accuracy={accuracy:.4f}, "
              f"AUC={auc:.4f}, "
              f"Loss={loss:.4f}")
    
    print("Federated training for Arbitrator Bias completed.")
    return state

if __name__ == '__main__':
    print("Running Arbitrator Bias Simulation (run_simulation_p3_task.py)...")
    final_state_arbitrator_bias = run_simulation_p3_arbitrator(num_rounds=5, num_clients=2, batch_size=16)
    print("\nArbitrator Bias Simulation finished.")

