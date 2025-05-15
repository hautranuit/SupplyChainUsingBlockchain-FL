import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import numpy as np
from data_preparation import make_federated_data, ELEMENT_SPEC, NUM_FEATURES
from federated_training import build_fed_avg_process
from model_definition import create_keras_model  # For loading final weights

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily.
nest_asyncio.apply()

def main():
    print("Starting Federated Learning Simulation for Sybil Detection...")
    
    # 1. Data Preparation
    NUM_CLIENTS_SIMULATION = 3
    CLIENT_IDS_SIMULATION = [f"sim_client_{i}" for i in range(NUM_CLIENTS_SIMULATION)]
    print(f"Preparing data for {NUM_CLIENTS_SIMULATION} clients...")
    
    # Each element in federated_train_data is a tf.data.Dataset for one client
    # These datasets should be preprocessed and batched appropriately for the model_fn
    # For TFF's from_keras_model, the dataset should yield (features, labels) tuples
    # where features and labels are for ONE batch.
    # Our load_local_data_for_client currently returns a dataset of individual examples.
    # We need to batch them before passing to iterative_process.next
    BATCH_SIZE = 32  # Define a batch size for client datasets
    raw_client_datasets = make_federated_data(CLIENT_IDS_SIMULATION, num_samples_per_client=200)
    
    def preprocess_client_dataset(dataset):
        # Shuffle and batch the client's dataset
        return dataset.shuffle(buffer_size=100).batch(BATCH_SIZE)
    
    federated_train_datasets = [preprocess_client_dataset(ds) for ds in raw_client_datasets]
    print("Client datasets prepared and batched.")
    
    # 2. Build Federated Training Process
    print("Building the federated training process (FedAvg)...")
    iterative_process = build_fed_avg_process()
    print("Federated training process built.")
    
    # 3. Initialize the Process
    print("Initializing the iterative process...")
    server_state = iterative_process.initialize()
    print("Initialization complete.")
    
    # 4. Run Federated Training Rounds
    NUM_ROUNDS = 10  # More rounds for better convergence
    print(f"Starting {NUM_ROUNDS} rounds of federated training...")
    
    for round_num in range(1, NUM_ROUNDS + 1):
        # Select a subset of clients for the round (here, using all for simplicity)
        # In a real system, client_data would be sampled from available clients.
        # The structure of client_data for iterative_process.next should be a list of client datasets.
        result = iterative_process.next(server_state, federated_train_datasets)  # Pass the list of datasets
        server_state = result.state
        metrics = result.metrics
        
        # The metrics structure depends on what's aggregated.
        # For FedAvg with Keras model, it's usually under 'client_work' then 'train'.
        round_loss = metrics['client_work']['train']['loss']
        round_accuracy = metrics['client_work']['train']['accuracy']  # if accuracy is a metric
        print(f"Round {round_num:2d}: loss={round_loss:.4f}, accuracy={round_accuracy:.4f}")
    
    print("Federated training completed.")
    
    # 5. Extract and Use the Global Model (Example)
    print("Extracting final global model weights...")
    model_weights = iterative_process.get_model_weights(server_state)
    
    # Create a new Keras model instance and assign the learned weights
    final_keras_model = create_keras_model()  # from model_definition.py
    final_keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Not strictly needed for inference but good practice
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    
    # Assign weights to the Keras model
    model_weights.assign_weights_to(final_keras_model)
    print("Final global model weights assigned to a new Keras model instance.")
    
    # Example: Evaluate the global model on some test data (simulated here)
    # In a real scenario, you'd have a separate federated_eval_data or a centralized test set.
    print("Simulating evaluation of the global model...")
    eval_data_X = np.random.rand(50, NUM_FEATURES).astype(np.float32)
    eval_data_y = np.random.randint(0, 2, size=(50, 1)).astype(np.int32)
    eval_results = final_keras_model.evaluate(eval_data_X, eval_data_y, verbose=0)
    print(f"Global model evaluation on simulated test data - Loss: {eval_results[0]:.4f}, "
          f"Accuracy: {eval_results[1]:.4f}, AUC: {eval_results[2]:.4f}")
    
    # Here, you would save final_keras_model or its weights for the Admin Dashboard Prediction Service
    # final_keras_model.save("global_sybil_detection_model.h5")
    # print("Global Keras model saved to global_sybil_detection_model.h5")
    
    print("\nSimulation finished.")

if __name__ == '__main__':
    main()