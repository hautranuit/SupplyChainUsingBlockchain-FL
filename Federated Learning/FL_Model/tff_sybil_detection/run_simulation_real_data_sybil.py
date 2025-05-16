import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import numpy as np

# Updated to use real data preparation for Sybil detection
from real_data_preparation_sybil import make_federated_data_sybil_real, ELEMENT_SPEC_SYBIL, NUM_SYBIL_FEATURES

from federated_training import build_fed_avg_process
from model_definition import create_keras_model # For loading final weights

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily.
nest_asyncio.apply()

def main():
    print("Starting Federated Learning Simulation for Sybil Detection with REAL DATA...")
    
    # 1. Data Preparation using real_data_preparation_sybil.py
    NUM_CLIENTS_SIMULATION = 2 # Adjust as needed, ensure enough nodes for distribution
    # Provide actual or placeholder node addresses for testing
    # These addresses should ideally exist and have some activity on the target blockchain (Amoy testnet)
    # For a robust test, these should be distinct and representative.
    EXAMPLE_NODE_ADDRESSES = [
        "0x1234567890123456789012345678901234567890", # Placeholder 1
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", # Placeholder 2
        "0xfedcba9876543210fedcba9876543210fedcba98"  # Placeholder 3
        # Add more addresses if NUM_CLIENTS_SIMULATION is larger or for better data distribution
    ]
    print(f"Preparing real data for {NUM_CLIENTS_SIMULATION} clients using addresses: {EXAMPLE_NODE_ADDRESSES[:NUM_CLIENTS_SIMULATION]}...")
    
    # ELEMENT_SPEC_SYBIL and NUM_SYBIL_FEATURES are imported from real_data_preparation_sybil
    # The make_federated_data_sybil_real function already handles client data distribution and batching (or implies it)
    # The `real_data_preparation_sybil.py` returns datasets already batched with batch_size=1.
    # If a different batch size is desired for training, it should be handled consistently.
    # For now, we assume the batching within make_federated_data_sybil_real is appropriate or will be adjusted if needed.
    federated_train_datasets = make_federated_data_sybil_real(EXAMPLE_NODE_ADDRESSES, num_fl_clients=NUM_CLIENTS_SIMULATION)
    print("Client datasets with real data prepared.")

    # Check if any client dataset is empty, which can happen if no nodes are assigned or no data is found
    if not federated_train_datasets or all(ds.cardinality().numpy() == 0 for ds in federated_train_datasets):
        print("Error: All client datasets are empty. This could be due to issues with node addresses, RPC connection, or no data found on-chain for these nodes.")
        print("Please ensure your .env file (ifps_qr.env) is correctly configured for blockchain_connector.py and that the example node addresses have activity.")
        print("Aborting simulation.")
        return
    
    # Filter out empty datasets to prevent errors in TFF process
    active_datasets = [ds for ds in federated_train_datasets if ds.cardinality().numpy() > 0]
    if not active_datasets:
        print("Error: No active (non-empty) client datasets available for training. Aborting simulation.")
        return
    if len(active_datasets) < len(federated_train_datasets):
        print(f"Warning: {len(federated_train_datasets) - len(active_datasets)} client(s) had no data and were excluded from this training round.")
    
    federated_train_datasets = active_datasets

    # 2. Build Federated Training Process
    print("Building the federated training process (FedAvg)...")
    # The model_fn inside build_fed_avg_process uses create_keras_model which expects NUM_SYBIL_FEATURES
    iterative_process = build_fed_avg_process(num_features=NUM_SYBIL_FEATURES, element_spec=ELEMENT_SPEC_SYBIL)
    print("Federated training process built.")
    
    # 3. Initialize the Process
    print("Initializing the iterative process...")
    server_state = iterative_process.initialize()
    print("Initialization complete.")
    
    # 4. Run Federated Training Rounds
    NUM_ROUNDS = 5  # Fewer rounds for a quick test, increase for actual training
    print(f"Starting {NUM_ROUNDS} rounds of federated training...")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        result = iterative_process.next(server_state, federated_train_datasets) 
        server_state = result.state
        metrics = result.metrics
        
        round_loss = metrics["client_work"]["train"]["loss"]
        round_accuracy = metrics["client_work"]["train"]["accuracy"]
        print(f"Round {round_num:2d}: loss={round_loss:.4f}, accuracy={round_accuracy:.4f}")
    
    print("Federated training completed.")
    
    # 5. Extract and Use the Global Model
    print("Extracting final global model weights...")
    model_weights = iterative_process.get_model_weights(server_state)
    
    final_keras_model = create_keras_model(num_features=NUM_SYBIL_FEATURES)
    final_keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    model_weights.assign_weights_to(final_keras_model)
    print("Final global model weights assigned to a new Keras model instance.")
    
    # Example: Evaluate the global model on a small sample from the first client's data (if available)
    print("Simulating evaluation of the global model on a sample of client data...")
    if federated_train_datasets and federated_train_datasets[0].cardinality().numpy() > 0:
        sample_client_data = list(federated_train_datasets[0].take(1))[0] # Take one batch
        eval_results = final_keras_model.evaluate(sample_client_data[0], sample_client_data[1], verbose=0)
        print(f"Global model evaluation on a sample batch - Loss: {eval_results[0]:.4f}, "
              f"Accuracy: {eval_results[1]:.4f}, AUC: {eval_results[2]:.4f}")
    else:
        print("No data available for sample evaluation.")
        
    # final_keras_model.save("/home/ubuntu/fl_integration_workspace/global_sybil_detection_model_real_data.h5")
    # print("Global Keras model (trained on real data) saved.")
    
    print("\nReal Data Simulation for Sybil Detection finished.")

if __name__ == '__main__':
    main()

