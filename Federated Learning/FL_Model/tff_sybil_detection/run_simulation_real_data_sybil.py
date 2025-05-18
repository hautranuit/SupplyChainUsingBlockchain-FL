import tensorflow_federated as tff
import tensorflow as tf
import nest_asyncio
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import time
from datetime import datetime

# Import custom modules
from real_data_preparation_sybil import make_federated_data_sybil_real, ELEMENT_SPEC_SYBIL, NUM_SYBIL_FEATURES
from federated_training import build_fed_avg_process
from model_definition import create_keras_model
from sybil_detection_model import SybilDetectionModel
from metrics_logger import MetricsLogger

# Apply nest_asyncio to allow TFF to run in environments like Jupyter or scripts easily
nest_asyncio.apply()

def main():
    print("Starting Improved Federated Learning Simulation for Sybil Detection with REAL DATA...")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "fl_integration")
    model_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results", f"fl_run_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"global_sybil_detection_model_{timestamp}.h5")
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(output_dir=results_dir)
    
    # Save experiment configuration
    config = {
        "experiment_name": "Sybil Detection with Real Data",
        "timestamp": timestamp,
        "num_clients": 4,
        "num_rounds": 20,
        "batch_size": 1,
        "learning_rate": 0.01,
        "model_type": "SybilDetectionModel",
        "features": NUM_SYBIL_FEATURES,
        "validation_frequency": 5
    }
    metrics_logger.save_experiment_config(config)
    
    # 1. Data Preparation using real_data_preparation_sybil.py
    NUM_CLIENTS_SIMULATION = config["num_clients"]
    EXAMPLE_NODE_ADDRESSES = [
        "0x5C6fF29A0f75E9d0dffC4374f600224EDc114449",  # Contract address
        "0x032041b4b356fEE1496805DD4749f181bC736FFA",  
        "0x04351e7dF40d04B5E610c4aA033faCf435b98711",  
        "0x5503a5B847e98B621d97695edf1bD84242C5862E",  
        "0x34Fc023EE50781e0a007852eEDC4A17fa353a8cD",  
        "0x724876f86fA52568aBc51955BD3A68bFc1441097",  
        "0x72EB9742d3B684ebA40F11573b733Ac9dB499f23", 
        "0x94081502540FD333075f3290d1D5C10A21AC5A5C"   
    ]
    print(f"Preparing real data for {NUM_CLIENTS_SIMULATION} clients using addresses: {EXAMPLE_NODE_ADDRESSES[:NUM_CLIENTS_SIMULATION]}...")
    
    start_time = time.time()
    federated_train_datasets = make_federated_data_sybil_real(EXAMPLE_NODE_ADDRESSES, num_fl_clients=NUM_CLIENTS_SIMULATION)
    data_prep_time = time.time() - start_time
    print(f"Client datasets with real data prepared in {data_prep_time:.2f} seconds.")

    # Check if any client dataset is empty
    if not federated_train_datasets or all(ds.cardinality().numpy() == 0 for ds in federated_train_datasets):
        print("Error: All client datasets are empty. This could be due to issues with node addresses, RPC connection, or no data found on-chain for these nodes.")
        print("Please ensure your .env file (ifps_qr.env) is correctly configured for blockchain_connector.py and that the example node addresses have activity.")
        print("Aborting simulation.")
        return
    
    # Filter out empty datasets
    active_datasets = [ds for ds in federated_train_datasets if ds.cardinality().numpy() > 0]
    if not active_datasets:
        print("Error: No active (non-empty) client datasets available for training. Aborting simulation.")
        return
    if len(active_datasets) < len(federated_train_datasets):
        print(f"Warning: {len(federated_train_datasets) - len(active_datasets)} client(s) had no data and were excluded from this training round.")
    
    federated_train_datasets = active_datasets
    
    # Log dataset statistics
    print("\nDataset Statistics:")
    all_features = []
    all_labels = []
    client_data_stats = {}
    
    for i, client_dataset in enumerate(federated_train_datasets):
        client_id = f"client_{i}"
        client_features = []
        client_labels = []
        
        for features_batch, labels_batch in client_dataset:
            for j in range(features_batch.shape[0]):
                client_features.append(features_batch[j].numpy())
                client_labels.append(labels_batch[j].numpy()[0])
        
        all_features.extend(client_features)
        all_labels.extend(client_labels)
        
        num_samples = len(client_labels)
        num_sybil = sum(client_labels)
        num_normal = num_samples - num_sybil
        
        client_data_stats[client_id] = {
            "num_samples": num_samples,
            "num_sybil": int(num_sybil),
            "num_normal": int(num_normal),
            "sybil_percentage": float(num_sybil / num_samples * 100) if num_samples > 0 else 0
        }
        
        print(f"  Client {i}: {num_samples} samples ({num_sybil} Sybil, {num_normal} Normal, {num_sybil / num_samples * 100:.1f}% Sybil)")
    
    total_samples = len(all_labels)
    total_sybil = sum(all_labels)
    total_normal = total_samples - total_sybil
    
    print(f"\nTotal: {total_samples} samples ({total_sybil} Sybil, {total_normal} Normal, {total_sybil / total_samples * 100:.1f}% Sybil)")
    
    # Save dataset statistics
    dataset_stats = {
        "total_samples": total_samples,
        "total_sybil": int(total_sybil),
        "total_normal": int(total_normal),
        "sybil_percentage": float(total_sybil / total_samples * 100) if total_samples > 0 else 0,
        "clients": client_data_stats
    }
    
    with open(os.path.join(results_dir, "dataset_statistics.json"), 'w') as f:
        import json
        json.dump(dataset_stats, f, indent=2)

    # 2. Build Federated Training Process
    print("\nBuilding the federated training process (FedAvg)...")
    start_time = time.time()
    iterative_process = build_fed_avg_process()
    build_time = time.time() - start_time
    print(f"Federated training process built in {build_time:.2f} seconds.")
    
    # 3. Initialize the Process
    print("Initializing the iterative process...")
    start_time = time.time()
    server_state = iterative_process.initialize()
    init_time = time.time() - start_time
    print(f"Initialization complete in {init_time:.2f} seconds.")
    
    # 4. Run Federated Training Rounds
    NUM_ROUNDS = config["num_rounds"]
    print(f"\nStarting {NUM_ROUNDS} rounds of federated training...")
    
    # Create validation model
    validation_model_wrapper = SybilDetectionModel()
    temp_keras_model = validation_model_wrapper.model
    
    # Prepare for collecting all predictions for final evaluation
    all_features_array = np.array(all_features)
    all_labels_array = np.array(all_labels)
    
    # Training loop
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}/{NUM_ROUNDS}:")
        
        # Train round
        start_time = time.time()
        result = iterative_process.next(server_state, federated_train_datasets)
        server_state = result.state
        metrics = result.metrics
        round_time = time.time() - start_time
        
        round_loss = metrics["client_work"]["train"]["loss"]
        round_accuracy = metrics["client_work"]["train"]["accuracy"]
        
        print(f"  Training - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f} (completed in {round_time:.2f} seconds)")
        
        # Log training metrics
        train_metrics = {
            "loss": float(round_loss),
            "accuracy": float(round_accuracy),
            "duration_seconds": round_time
        }
        metrics_logger.log_train_metrics(round_num, train_metrics)
        
        # Validation every N rounds
        if round_num % config["validation_frequency"] == 0 or round_num == NUM_ROUNDS:
            print("  Performing validation...")
            
            # Get current global model weights from server_state
            current_model_weights = iterative_process.get_model_weights(server_state)
            
            # Assign these weights to the temporary Keras model
            current_model_weights.assign_weights_to(temp_keras_model)
            
            # Collect validation metrics
            val_loss = 0
            val_accuracy = 0
            val_count = 0
            
            all_val_true = []
            all_val_pred = []
            
            start_time = time.time()
            
            for client_dataset in federated_train_datasets:
                for features_batch, labels_batch in client_dataset:
                    # Convert to numpy for easier handling
                    features_np = features_batch.numpy()
                    labels_np = labels_batch.numpy()
                    
                    # Make predictions
                    predictions = temp_keras_model.predict(features_np, verbose=0)
                    
                    # Collect true labels and predictions for metrics
                    all_val_true.extend(labels_np.flatten())
                    all_val_pred.extend(predictions.flatten())
                    
                    # Calculate batch metrics
                    batch_loss = tf.keras.losses.binary_crossentropy(
                        tf.cast(labels_np, dtype=tf.float32), 
                        predictions
                    ).numpy().mean()
                    
                    batch_accuracy = tf.keras.metrics.binary_accuracy(
                        tf.cast(labels_np, dtype=tf.float32),
                        predictions
                    ).numpy().mean()
                    
                    val_loss += batch_loss
                    val_accuracy += batch_accuracy
                    val_count += 1
            
            val_time = time.time() - start_time
            
            if val_count > 0:
                val_loss /= val_count
                val_accuracy /= val_count
                
                # Calculate ROC AUC
                try:
                    fpr, tpr, _ = roc_curve(all_val_true, all_val_pred)
                    val_auc = auc(fpr, tpr)
                except Exception as e:
                    print(f"  Warning: Could not calculate AUC: {e}")
                    val_auc = 0
                
                print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f} (completed in {val_time:.2f} seconds)")
                
                # Log validation metrics
                val_metrics = {
                    "loss": float(val_loss),
                    "accuracy": float(val_accuracy),
                    "auc": float(val_auc),
                    "duration_seconds": val_time
                }
                metrics_logger.log_validation_metrics(round_num, val_metrics)
                
                # Log confusion matrix
                metrics_logger.log_confusion_matrix(round_num, all_val_true, all_val_pred)
            
            # Plot ROC curve on final round
            if round_num == NUM_ROUNDS:
                metrics_logger.plot_roc_curve(all_val_true, all_val_pred)
                metrics_logger.plot_precision_recall_curve(all_val_true, all_val_pred)
    
    print("\nTraining completed.")
    
    # 5. Extract and Save the Global Model
    print("\nExtracting final global model weights...")
    model_weights = iterative_process.get_model_weights(server_state)
    
    # Create and save the model
    sybil_model = SybilDetectionModel()
    model_weights.assign_weights_to(sybil_model.model)
    sybil_model.save_model(model_path)
    print(f"Final global model saved to {model_path}")
    
    # 6. Final Evaluation
    print("\nPerforming final evaluation on all data...")
    
    # Make predictions on all data
    all_predictions = sybil_model.model.predict(all_features_array, verbose=0)
    
    # Calculate metrics
    test_loss = tf.keras.losses.binary_crossentropy(
        tf.cast(all_labels_array, dtype=tf.float32),
        all_predictions
    ).numpy().mean()
    
    test_accuracy = tf.keras.metrics.binary_accuracy(
        tf.cast(all_labels_array, dtype=tf.float32),
        all_predictions
    ).numpy().mean()
    
    # Calculate ROC AUC
    try:
        fpr, tpr, _ = roc_curve(all_labels_array, all_predictions)
        test_auc = auc(fpr, tpr)
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        test_auc = 0
    
    print(f"Final evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
    
    # Log test metrics
    test_metrics = {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "auc": float(test_auc)
    }
    metrics_logger.log_test_metrics(test_metrics)
    
    # Generate confusion matrix for final model
    metrics_logger.log_confusion_matrix(NUM_ROUNDS + 1, all_labels_array, all_predictions)
    
    # 7. Generate Summary Report and Plots
    print("\nGenerating summary report and plots...")
    metrics_logger.plot_training_curves()
    summary_report = metrics_logger.generate_summary_report()
    
    print("\nImproved Real Data Simulation for Sybil Detection finished.")
    print(f"All results saved to {results_dir}")
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
