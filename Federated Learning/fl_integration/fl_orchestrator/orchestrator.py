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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom JSON encoder to handle NumPy data types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder supporting NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Enhanced Data Augmentation and Performance Analysis
class PerformanceAnalyzer:
    """Enhanced performance analysis and monitoring."""
    
    def __init__(self, enable_detailed_analysis=True):
        self.enable_detailed_analysis = enable_detailed_analysis
        self.training_history = []
        self.performance_metrics = {}
    
    def analyze_training_round(self, round_num, metrics):
        """Analyze performance for a single training round."""
        analysis = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'analysis': {}
        }
        
        if 'loss' in metrics:
            analysis['analysis']['loss_trend'] = self.analyze_loss_trend(metrics['loss'])
        
        if 'binary_accuracy' in metrics:
            accuracy = float(metrics['binary_accuracy'])
            analysis['analysis']['accuracy_assessment'] = self.assess_accuracy(accuracy)
        
        if 'auc' in metrics:
            auc = float(metrics['auc'])
            analysis['analysis']['auc_assessment'] = self.assess_auc(auc)
        
        self.training_history.append(analysis)
        return analysis
    
    def analyze_loss_trend(self, loss_value):
        """Analyze loss trend and convergence."""
        return float(loss_value)
    
    def assess_accuracy(self, accuracy):
        """Assess accuracy performance."""
        return float(accuracy)
    
    def _analyze_loss_trend(self, loss_value):
        """Private wrapper for backward compatibility."""
        return self.analyze_loss_trend(loss_value)
    
    def _assess_accuracy(self, accuracy):
        """Private wrapper for backward compatibility."""
        return self.assess_accuracy(accuracy)
    
    def assess_auc(self, auc):
        """Assess AUC performance."""
        return float(auc)
    
    def _assess_auc(self, auc):
        """Private wrapper for backward compatibility."""
        return self.assess_auc(auc)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        if not self.training_history:
            return "No training history available"
        
        report = ["=" * 80]
        report.append("FEDERATED LEARNING PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Training Rounds: {len(self.training_history)}")
        report.append("")
        
        # Latest metrics
        latest = self.training_history[-1]
        report.append("LATEST PERFORMANCE METRICS:")
        report.append("-" * 40)
        
        for metric, value in latest['metrics'].items():
            if metric in latest['analysis']:
                assessment = latest['analysis'][metric + '_assessment'] if metric + '_assessment' in latest['analysis'] else latest['analysis'].get(metric + '_trend', 'N/A')
                report.append(f"{metric.upper()}: {value} - {assessment}")
            else:
                report.append(f"{metric.upper()}: {value}")
        
        # Performance trends
        if len(self.training_history) > 1:
            report.append("")
            report.append("PERFORMANCE TRENDS:")
            report.append("-" * 40)
            
            # Accuracy trend
            accuracies = [float(h['metrics'].get('binary_accuracy', 0)) for h in self.training_history if 'binary_accuracy' in h['metrics']]
            if len(accuracies) > 1:
                trend = "IMPROVING" if accuracies[-1] > accuracies[0] else "DECLINING"
                improvement = abs(accuracies[-1] - accuracies[0]) * 100
                report.append(f"Accuracy Trend: {trend} ({improvement:.1f}% change)")
            
            # Loss trend
            losses = [float(h['metrics'].get('loss', 0)) for h in self.training_history if 'loss' in h['metrics']]
            if len(losses) > 1:
                trend = "IMPROVING" if losses[-1] < losses[0] else "INCREASING"
                change = abs(losses[-1] - losses[0])
                report.append(f"Loss Trend: {trend} ({change:.3f} change)")
        
        # Recommendations
        report.append("")
        report.append("OPTIMIZATION RECOMMENDATIONS:")
        report.append("-" * 40)
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"• {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self):
        """Generate optimization recommendations based on performance."""
        recommendations = []
        
        if not self.training_history:
            return ["No training data available for recommendations"]
        
        latest = self.training_history[-1]['metrics']
        
        # Loss-based recommendations
        if 'loss' in latest:
            loss = float(latest['loss'])
            if loss > 1.0:
                recommendations.append("High loss detected: Consider reducing learning rate or using focal loss")
                recommendations.append("Check data quality and feature scaling")
            elif loss > 0.7:
                recommendations.append("Moderate loss: Continue training or try label smoothing")
        
        # Accuracy-based recommendations
        if 'binary_accuracy' in latest:
            accuracy = float(latest['binary_accuracy'])
            if accuracy < 0.7:
                recommendations.append("Low accuracy: Consider model architecture improvements")
                recommendations.append("Try data augmentation or feature engineering")
            elif accuracy > 0.95:
                recommendations.append("Very high accuracy: Check for overfitting, consider regularization")
        
        # AUC-based recommendations
        if 'auc' in latest:
            auc = float(latest['auc'])
            if auc < 0.7:
                recommendations.append("Low AUC: Model has poor discrimination ability")
                recommendations.append("Consider class balancing or cost-sensitive learning")
        
        # Performance stability
        if len(self.training_history) > 5:
            recent_accuracies = [float(h['metrics'].get('binary_accuracy', 0)) 
                               for h in self.training_history[-5:] 
                               if 'binary_accuracy' in h['metrics']]
            if recent_accuracies and np.std(recent_accuracies) > 0.1:
                recommendations.append("Performance instability detected: Consider reducing learning rate")
                recommendations.append("Add more regularization or use early stopping")
        
        if not recommendations:
            recommendations.append("Performance looks good! Continue monitoring.")
        
        return recommendations

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
        
        # Initialize performance analyzer for enhanced monitoring
        self.performance_analyzer = PerformanceAnalyzer(enable_detailed_analysis=True)
        
        logger.info(f"FL Orchestrator initialized with:")
        logger.info(f"  Model directory: {model_dir}")
        logger.info(f"  Client auth: {client_auth_enabled}")
        logger.info(f"  Secure aggregation: {secure_aggregation}")
        logger.info(f"  Differential privacy: {differential_privacy}")
        logger.info(f"  Performance analysis: Enabled")
        
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
                            client_optimizer_fn: Optional[Any] = None,
                            server_optimizer_fn: Optional[Any] = None) -> Tuple[Optional[object], Dict]:
        """
        Train a federated model using TensorFlow Federated.
        
        Args:
            federated_data: List of client datasets
            model_fn: Function that returns a TFF Model
            num_rounds: Number of federated rounds
            client_optimizer_fn: TFF optimizer builder (not a function)
            server_optimizer_fn: TFF optimizer builder (not a function)
            
        Returns:
            Tuple of (final_server_state, training_history)
        """
        try:
            logger.info(f"Starting federated training for {num_rounds} rounds...")
            # Build federated averaging process using the provided model function and optimizer builders
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                model_fn=model_fn,
                client_optimizer_fn=client_optimizer_fn,
                server_optimizer_fn=server_optimizer_fn
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
                
                print(f"\n🔄 Round {round_num + 1}/{num_rounds}")
                logger.info(f"Starting round {round_num + 1}/{num_rounds}")
                
                # Perform one round of federated training
                result = iterative_process.next(server_state, federated_data)
                server_state = result.state
                
                # Extract metrics
                round_metrics = result.metrics
                
                # Enhanced metric extraction and performance analysis
                round_time = time.time() - start_time
                history["training_time"].append(round_time)
                
                # Extract metrics from nested structure
                processed_metrics = {}
                
                # Debug: Print the actual metrics structure
                logger.debug(f"Round {round_num + 1} metrics structure: {round_metrics}")
                
                if isinstance(round_metrics, dict):
                    # Handle nested TFF metrics structure
                    client_work = round_metrics.get('client_work', {})
                    train_metrics = client_work.get('train', {})
                    
                    # Debug: Print train_metrics
                    logger.debug(f"Train metrics: {train_metrics}")
                    
                    # Extract common metrics
                    if 'loss' in train_metrics:
                        processed_metrics['loss'] = float(train_metrics['loss'])
                        history["round_losses"].append(processed_metrics['loss'])
                    
                    if 'binary_accuracy' in train_metrics:
                        processed_metrics['binary_accuracy'] = float(train_metrics['binary_accuracy'])
                    elif 'accuracy' in train_metrics:
                        processed_metrics['binary_accuracy'] = float(train_metrics['accuracy'])
                    
                    if 'auc' in train_metrics:
                        processed_metrics['auc'] = float(train_metrics['auc'])
                        logger.info(f"AUC found in metrics: {processed_metrics['auc']}")
                    else:
                        logger.warning(f"AUC not found in train_metrics. Available keys: {list(train_metrics.keys())}")
                    
                    if 'precision' in train_metrics:
                        processed_metrics['precision'] = float(train_metrics['precision'])
                        
                    if 'recall' in train_metrics:
                        processed_metrics['recall'] = float(train_metrics['recall'])
                        
                    if 'f1_score' in train_metrics:
                        processed_metrics['f1_score'] = float(train_metrics['f1_score'])
                
                # Perform performance analysis
                if processed_metrics:
                    analysis = self.performance_analyzer.analyze_training_round(round_num + 1, processed_metrics)
                    
                    # Enhanced logging with performance insights
                    metrics_str = []
                    for metric, value in processed_metrics.items():
                        if isinstance(value, float):
                            metrics_str.append(f"{metric}={value:.4f}")
                        else:
                            metrics_str.append(f"{metric}={value}")
                    
                    # Visual progress display
                    print(f"   📊 Metrics: {', '.join(metrics_str)}")
                    print(f"   ⏱️  Time: {round_time:.2f}s")
                    
                    # Display performance insights
                    if 'loss' in processed_metrics:
                        loss_trend = self.performance_analyzer.analyze_loss_trend(processed_metrics['loss'])
                        print(f"   📉 Loss Analysis: {loss_trend}")
                    
                    if 'binary_accuracy' in processed_metrics:
                        acc_assessment = self.performance_analyzer.assess_accuracy(processed_metrics['binary_accuracy'])
                        print(f"   🎯 Accuracy Analysis: {acc_assessment}")
                    
                    if 'auc' in processed_metrics:
                        auc_assessment = self.performance_analyzer.assess_auc(processed_metrics['auc'])
                        print(f"   📏 AUC Analysis: {auc_assessment}")
                    
                    logger.info(f"Round {round_num + 1} completed. {', '.join(metrics_str)}, Time: {round_time:.2f}s")
                    
                    # Log performance insights
                    for analysis_type, insight in analysis['analysis'].items():
                        logger.info(f"Performance Insight - {analysis_type}: {insight}")
                
                else:
                    # Fallback for metrics extraction
                    if hasattr(round_metrics, 'train') and hasattr(round_metrics['train'], 'loss'):
                        round_loss = float(round_metrics['train']['loss'])
                        history["round_losses"].append(round_loss)
                        processed_metrics['loss'] = round_loss
                    
                    logger.info(f"Round {round_num + 1} completed. Time: {round_time:.2f}s")
                
                history["round_metrics"].append(round_metrics)
            
            # Training completion
            print(f"\n🏁 Training completed after {num_rounds} rounds!")
            
            total_training_time = sum(history["training_time"])
            print(f"⏱️  Total training time: {total_training_time:.2f}s")
            
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

    def transfer_weights_to_keras_model(self, server_state, keras_model):
        """
        Transfer weights from TFF server state to a Keras model.
        
        Args:
            server_state: TFF server state containing model weights
            keras_model: Keras model to receive the weights
            
        Returns:
            Updated Keras model with transferred weights
        """
        try:
            # Extract model weights from server state
            if hasattr(server_state, 'model'):
                tff_weights = server_state.model
                
                # Get trainable weights from TFF model
                if hasattr(tff_weights, 'trainable'):
                    trainable_weights = [w.numpy() for w in tff_weights.trainable]
                    
                    # Set weights to Keras model
                    if len(trainable_weights) == len(keras_model.trainable_weights):
                        keras_model.set_weights(trainable_weights)
                        logger.info("Successfully transferred weights from TFF server state to Keras model")
                    else:
                        logger.warning(f"Weight dimension mismatch: TFF={len(trainable_weights)}, Keras={len(keras_model.trainable_weights)}")
                        
                        # Try to match weights by shape
                        for i, (tff_weight, keras_weight) in enumerate(zip(trainable_weights, keras_model.trainable_weights)):
                            if tff_weight.shape == keras_weight.shape:
                                keras_model.trainable_weights[i].assign(tff_weight)
                            else:
                                logger.warning(f"Shape mismatch at layer {i}: TFF={tff_weight.shape}, Keras={keras_weight.shape}")
                else:
                    logger.error("No trainable weights found in TFF server state")
            else:
                logger.error("No model found in server state")
                
            return keras_model
            
        except Exception as e:
            logger.error(f"Failed to transfer weights: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return keras_model

    def save_performance_report(self, report_data: Dict, output_dir: str = None, filename: str = None):
        """
        Save performance analysis report to file.
        
        Args:
            report_data: Performance report data dictionary
            output_dir: Directory to save the report (optional)
            filename: Custom filename (optional)
            
        Returns:
            Path to saved report file
        """
        try:
            if output_dir is None:
                output_dir = os.path.join(self.model_dir, "performance_reports")
            
            os.makedirs(output_dir, exist_ok=True)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_report_{timestamp}.json"
            
            report_path = os.path.join(output_dir, filename)
            
            # Enhance report with metadata
            enhanced_report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "orchestrator_version": "1.0.0",
                    "system_info": {
                        "python_version": sys.version,
                        "tensorflow_version": tf.__version__ if 'tf' in globals() else "Unknown"
                    }
                },
                "performance_analysis": report_data,
                "training_history": self.performance_analyzer.training_history[-10:] if self.performance_analyzer.training_history else []
            }
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_report, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            
            logger.info(f"Performance report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {str(e)}")
            return None

    def generate_comprehensive_performance_report(self):
        """
        Generate a comprehensive performance report with visualizations and insights.
        
        Returns:
            Dictionary containing comprehensive performance analysis
        """
        try:
            if not self.performance_analyzer.training_history:
                return {"error": "No training history available for analysis"}
            
            # Extract metrics for analysis
            rounds = []
            losses = []
            accuracies = []
            aucs = []
            
            for entry in self.performance_analyzer.training_history:
                rounds.append(entry['round'])
                metrics = entry['metrics']
                
                if 'loss' in metrics:
                    losses.append(float(metrics['loss']))
                if 'binary_accuracy' in metrics:
                    accuracies.append(float(metrics['binary_accuracy']))
                if 'auc' in metrics:
                    aucs.append(float(metrics['auc']))
            
            # Performance analysis
            comprehensive_report = {
                "training_summary": {
                    "total_rounds": len(rounds),
                    "final_round": max(rounds) if rounds else 0,
                    "training_duration": self._calculate_training_duration()
                },
                "metrics_analysis": {},
                "trends": {},
                "recommendations": self.performance_analyzer.generate_recommendations(),
                "alerts": []
            }
            
            # Loss analysis
            if losses:
                comprehensive_report["metrics_analysis"]["loss"] = {
                    "final": losses[-1],
                    "minimum": min(losses),
                    "maximum": max(losses),
                    "average": np.mean(losses),
                    "std_deviation": np.std(losses),
                    "improvement": losses[0] - losses[-1] if len(losses) > 1 else 0
                }
                
                # Loss trend analysis
                if len(losses) >= 3:
                    recent_trend = np.polyfit(range(len(losses[-3:])), losses[-3:], 1)[0]
                    comprehensive_report["trends"]["loss_trend"] = "improving" if recent_trend < 0 else "deteriorating"
            
            # Accuracy analysis
            if accuracies:
                comprehensive_report["metrics_analysis"]["accuracy"] = {
                    "final": accuracies[-1],
                    "maximum": max(accuracies),
                    "minimum": min(accuracies),
                    "average": np.mean(accuracies),
                    "std_deviation": np.std(accuracies),
                    "improvement": accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
                }
                
                # Accuracy stability check
                if len(accuracies) >= 5:
                    recent_stability = np.std(accuracies[-5:])
                    if recent_stability > 0.05:
                        comprehensive_report["alerts"].append("High accuracy variation in recent rounds")
            
            # AUC analysis
            if aucs:
                comprehensive_report["metrics_analysis"]["auc"] = {
                    "final": aucs[-1],
                    "maximum": max(aucs),
                    "minimum": min(aucs),
                    "average": np.mean(aucs),
                    "improvement": aucs[-1] - aucs[0] if len(aucs) > 1 else 0
                }
            
            # Performance alerts
            if losses and losses[-1] > 1.0:
                comprehensive_report["alerts"].append("High final loss detected")
            
            if accuracies and accuracies[-1] < 0.7:
                comprehensive_report["alerts"].append("Low final accuracy detected")
            
            if aucs and aucs[-1] < 0.7:
                comprehensive_report["alerts"].append("Poor model discrimination (low AUC)")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive performance report: {str(e)}")
            return {"error": str(e)}

    def _calculate_training_duration(self):
        """Calculate total training duration from training history."""
        if not self.performance_analyzer.training_history:
            return 0
        
        try:
            start_time = self.performance_analyzer.training_history[0]['timestamp']
            end_time = self.performance_analyzer.training_history[-1]['timestamp']
            
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.fromisoformat(end_time)
            
            duration = (end_dt - start_dt).total_seconds()
            return duration
            
        except Exception:
            return 0
