import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import numpy as np
from data_preparation_phase2 import make_federated_data_phase2, ELEMENT_SPEC_PHASE2, NUM_PHASE2_FEATURES
from federated_training import build_weighted_fed_avg
from model_definition import tff_model_fn

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily.
nest_asyncio.apply()

def preprocess_client_dataset(dataset):
    """Preprocesses a client dataset for training."""
    # The dataset is already batched from data_preparation_phase2
    return dataset

def main():
    print("Starting Federated Learning Simulation for Batch Processing Anomaly Detection...")
    
    # Prepare data for Phase 2
    print("Preparing data for Phase 2 clients...")
    federated_train_datasets = make_federated_data_phase2(num_fl_clients=3)
    
    # Preprocess each client's dataset
    federated_train_datasets = [preprocess_client_dataset(ds) for ds in federated_train_datasets]
    print("Client datasets prepared and batched.")
    
    # Build the federated training process
    print("Building the federated training process (FedAvg)...")
    iterative_process = build_weighted_fed_avg(
        model_fn=tff_model_fn,
        client_learning_rate=0.1,
        server_learning_rate=1.0
    )
    print("Federated training process built.")
    
    # Initialize the process
    print("Initializing the iterative process...")
    server_state = iterative_process.initialize()
    print("Initialization complete.")
    
    # Run federated training
    print("Starting 10 rounds of federated training...")
    for round_num in range(10):
        result = iterative_process.next(server_state, federated_train_datasets)
        server_state = result.state
        train_metrics = result.metrics
        
        print(f"Round {round_num + 1:2d}: "
              f"loss={train_metrics['loss']:.4f}, "
              f"accuracy={train_metrics['accuracy']:.4f}")
    
    print("Federated training completed.")
    
    # Extract the final model weights
    print("Extracting final global model weights...")
    keras_model = tff_model_fn()
    keras_model.load_weights(server_state.model_weights)
    print("Final global model weights assigned to a new Keras model instance.")
    
    # Simulate evaluation
    print("Simulating evaluation of the global model...")
    # Create a small test dataset using the same data preparation
    test_datasets = make_federated_data_phase2(num_fl_clients=1)
    test_dataset = preprocess_client_dataset(test_datasets[0])
    
    # Evaluate the model
    evaluation_metrics = keras_model.evaluate(test_dataset)
    print(f"Global model evaluation on simulated test data - "
          f"Loss: {evaluation_metrics['loss']:.4f}, "
          f"Accuracy: {evaluation_metrics['accuracy']:.4f}, "
          f"AUC: {evaluation_metrics['auc']:.4f}")
    
    print("\nSimulation finished.")

if __name__ == '__main__':
    main()