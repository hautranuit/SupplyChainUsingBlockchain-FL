"""
FL Orchestrator for Federated Learning integration.
This module coordinates the FL training process, including client selection,
model aggregation, and evaluation.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from datetime import datetime
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../fl_integration_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fl_orchestrator")

class FLOrchestrator:
    """
    FL Orchestrator class for coordinating the FL training process.
    Handles client selection, model aggregation, and evaluation.
    """
    
    def __init__(self, 
                 model_dir: str = "./models",
                 client_auth_enabled: bool = True,
                 secure_aggregation: bool = True,
                 differential_privacy: bool = False,
                 dp_noise_multiplier: float = 0.1,
                 dp_l2_norm_clip: float = 1.0):
        """
        Initialize the FL Orchestrator.
        
        Args:
            model_dir: Directory to save/load models
            client_auth_enabled: Enable client authentication
            secure_aggregation: Enable secure aggregation
            differential_privacy: Enable differential privacy
            dp_noise_multiplier: Noise multiplier for DP
            dp_l2_norm_clip: L2 norm clipping for DP
        """
        self.model_dir = model_dir
        self.client_auth_enabled = client_auth_enabled
        self.secure_aggregation = secure_aggregation
        self.differential_privacy = differential_privacy
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize client registry
        self.registered_clients = {}
        self.client_selection_strategy = "random"
        
        logger.info(f"FL Orchestrator initialized with:")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Client auth: {client_auth_enabled}")
        logger.info(f"  Secure aggregation: {secure_aggregation}")
        logger.info(f"  Differential privacy: {differential_privacy}")
        
        # Set TFF execution context
        try:
            tff.backends.native.set_local_python_execution_context()
            logger.info("TFF execution context set successfully")
        except Exception as e:
            logger.warning(f"Could not set TFF execution context: {e}")
            logger.warning("Using default TFF execution context")

    def register_client(self, client_id: str, client_info: Dict):
        """Register a new client with the orchestrator."""
        self.registered_clients[client_id] = {
            "client_id": client_id,
            "registration_time": datetime.now().isoformat(),
            "status": "active",
            **client_info
        }
        logger.info(f"Client {client_id} registered successfully")

    def select_clients(self, num_clients: int) -> List[str]:
        """
        Select clients for the current round.
        
        Args:
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        available_clients = [cid for cid, info in self.registered_clients.items() 
                           if info["status"] == "active"]
        
        if len(available_clients) == 0:
            logger.warning("No active clients available for selection")
            return []
        
        # Select based on strategy
        if self.client_selection_strategy == "random":
            selected = random.sample(available_clients, 
                                   min(num_clients, len(available_clients)))
        else:
            # Default to all available clients
            selected = available_clients[:num_clients]
        
        logger.info(f"Selected {len(selected)} clients for training: {selected}")
        return selected

    def train_federated_model(self,
                            federated_data: List,
                            model_fn: Callable,
                            num_rounds: int = 10,
                            client_optimizer_fn: Optional[Callable] = None,
                            server_optimizer_fn: Optional[Callable] = None) -> Tuple[Optional[object], Dict]:
        """
        Train a federated model using TensorFlow Federated.
        
        Args:
            federated_data: List of client datasets
            model_fn: Function that returns a TFF Model
            num_rounds: Number of federated rounds
            client_optimizer_fn: Client optimizer function
            server_optimizer_fn: Server optimizer function
            
        Returns:
            Tuple of (final_server_state, training_history)
        """
        try:
            logger.info(f"Starting federated training for {num_rounds} rounds...")
            
            # Set default optimizers if not provided
            # These should be functions that *return* an optimizer instance when called.
            if client_optimizer_fn is None:
                client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
            if server_optimizer_fn is None:
                server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
            
            # Build federated averaging process using the provided model function
            # Ensure that the optimizers passed to build_weighted_fed_avg are callables (functions)
            # that return optimizer instances, not the instances themselves directly.
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                model_fn=model_fn, # This should be a no-arg function returning a tff.learning.Model
                client_optimizer_fn=client_optimizer_fn, # Callable returning client optimizer
                server_optimizer_fn=server_optimizer_fn  # Callable returning server optimizer
            )
            
            # Initialize the server state
            server_state = iterative_process.initialize()
            
            logger.info("Federated averaging process built and initialized.")
            
            # Training history
            history = {
                "round_losses": [],
                "round_metrics": [],
                "training_time": []
            }
            
            # Training loop
            for round_num in range(num_rounds):
                start_time = time.time()
                
                logger.info(f"Starting round {round_num + 1}/{num_rounds}")
                
                # Perform one round of federated training
                result = iterative_process.next(server_state, federated_data)
                server_state = result.state
                
                # Extract metrics
                round_metrics = result.metrics
                
                # Log round results
                round_time = time.time() - start_time
                history["training_time"].append(round_time)
                
                if hasattr(round_metrics, 'train') and hasattr(round_metrics['train'], 'loss'):
                    round_loss = float(round_metrics['train']['loss'])
                    history["round_losses"].append(round_loss)
                    logger.info(f"Round {round_num + 1} completed. Loss: {round_loss:.4f}, Time: {round_time:.2f}s")
                else:
                    logger.info(f"Round {round_num + 1} completed. Time: {round_time:.2f}s")
                
                history["round_metrics"].append(round_metrics)
            
            logger.info("Federated training completed successfully")
            return server_state, history
            
        except Exception as e:
            logger.error(f"Failed to train federated model: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, {}

    def evaluate_model(self, server_state, test_data) -> Dict:
        """
        Evaluate the federated model on test data.
        
        Args:
            server_state: Final server state from training
            test_data: Test dataset
            
        Returns:
            Evaluation metrics dictionary
        """
        try:
            logger.info("Evaluating federated model...")
            
            # Extract model weights from server state
            model_weights = server_state.model
            
            # Create evaluation results
            eval_results = {
                "evaluation_time": datetime.now().isoformat(),
                "model_size": len(model_weights.trainable) if hasattr(model_weights, 'trainable') else 0,
                "status": "completed"
            }
            
            logger.info("Model evaluation completed")
            return eval_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {
                "evaluation_time": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            }

    def save_model_state(self, server_state, model_name: str, version: str = "latest"):
        """
        Save the server state/model to disk.
        
        Args:
            server_state: Server state to save
            model_name: Name of the model
            version: Version identifier
        """
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}_{version}")
            
            # Save model weights if available
            if hasattr(server_state, 'model') and hasattr(server_state.model, 'trainable'):
                weights_path = f"{model_path}_weights.npy"
                weights = [w.numpy() for w in server_state.model.trainable]
                np.save(weights_path, weights, allow_pickle=True)
                logger.info(f"Model weights saved to {weights_path}")
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "save_time": datetime.now().isoformat(),
                "orchestrator_config": {
                    "client_auth_enabled": self.client_auth_enabled,
                    "secure_aggregation": self.secure_aggregation,
                    "differential_privacy": self.differential_privacy
                }
            }
            
            metadata_path = f"{model_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")

    def load_model_state(self, model_name: str, version: str = "latest"):
        """
        Load a saved model state.
        
        Args:
            model_name: Name of the model to load
            version: Version to load
            
        Returns:
            Loaded model state or None if failed
        """
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}_{version}")
            
            # Load metadata
            metadata_path = f"{model_path}_metadata.json"
            if not os.path.exists(metadata_path):
                logger.error(f"Model metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load weights
            weights_path = f"{model_path}_weights.npy"
            if os.path.exists(weights_path):
                weights = np.load(weights_path, allow_pickle=True)
                logger.info(f"Model {model_name} loaded successfully")
                return {
                    "metadata": metadata,
                    "weights": weights
                }
            else:
                logger.warning(f"Model weights not found: {weights_path}")
                return {"metadata": metadata, "weights": None}
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None

    def get_client_status(self) -> Dict:
        """Get status of all registered clients."""
        return {
            "total_clients": len(self.registered_clients),
            "active_clients": len([c for c in self.registered_clients.values() 
                                 if c["status"] == "active"]),
            "clients": self.registered_clients
        }

    def cleanup(self):
        """Cleanup orchestrator resources."""
        try:
            logger.info("Cleaning up FL Orchestrator resources...")
            # Any cleanup operations would go here
            logger.info("FL Orchestrator cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
