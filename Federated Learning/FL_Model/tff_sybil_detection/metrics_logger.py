import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

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

class MetricsLogger:
    """
    A class for logging, visualizing, and analyzing metrics during Federated Learning training.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the MetricsLogger.
        
        Args:
            output_dir: Directory to save metrics, plots, and reports. If None, creates a timestamped directory.
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "fl_integration", "results")
            os.makedirs(base_dir, exist_ok=True)
            self.output_dir = os.path.join(base_dir, f"fl_run_{timestamp}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.confusion_matrices = []
        self.client_metrics = {}
        
        # Initialize metrics file
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")
        self._initialize_metrics_file()
        
        print(f"MetricsLogger initialized. Output directory: {self.output_dir}")
    
    def _initialize_metrics_file(self):
        """Initialize the metrics JSON file with basic structure."""
        metrics_data = {
            "train": [],
            "validation": [],
            "test": [],
            "client_metrics": {},
            "confusion_matrices": [],
            "timestamp": datetime.now().isoformat(),
            "experiment_config": {}
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, cls=NumpyJSONEncoder)
    
    def log_train_metrics(self, round_num, metrics):
        """
        Log training metrics for a specific round.
        
        Args:
            round_num: The current training round number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        metrics["round"] = round_num
        metrics["timestamp"] = datetime.now().isoformat()
        self.train_metrics.append(metrics)
        
        # Update metrics file
        self._update_metrics_file("train", metrics)
        
        print(f"Round {round_num} training metrics logged.")
    
    def log_validation_metrics(self, round_num, metrics):
        """
        Log validation metrics for a specific round.
        
        Args:
            round_num: The current training round number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        metrics["round"] = round_num
        metrics["timestamp"] = datetime.now().isoformat()
        self.val_metrics.append(metrics)
        
        # Update metrics file
        self._update_metrics_file("validation", metrics)
        
        print(f"Round {round_num} validation metrics logged.")
    
    def log_test_metrics(self, metrics):
        """
        Log test metrics after training.
        
        Args:
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        metrics["timestamp"] = datetime.now().isoformat()
        self.test_metrics.append(metrics)
        
        # Update metrics file
        self._update_metrics_file("test", metrics)
        
        print("Test metrics logged.")
    
    def log_client_metrics(self, client_id, round_num, metrics):
        """
        Log metrics for a specific client in a specific round.
        
        Args:
            client_id: Identifier for the client
            round_num: The current training round number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
        """
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []
        
        metrics["round"] = round_num
        metrics["timestamp"] = datetime.now().isoformat()
        self.client_metrics[client_id].append(metrics)
        
        # Update metrics file
        self._update_metrics_file("client_metrics", {client_id: metrics})
        
        print(f"Client {client_id} metrics for round {round_num} logged.")
    
    def log_confusion_matrix(self, round_num, y_true, y_pred, threshold=0.5):
        """
        Compute and log confusion matrix for a specific round.
        
        Args:
            round_num: The current training round number
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Threshold for binary classification
        """
        # Convert probabilities to binary predictions
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary).tolist()
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        cm_data = {
            "round": round_num,
            "confusion_matrix": cm,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "timestamp": datetime.now().isoformat()
        }
        
        self.confusion_matrices.append(cm_data)
        
        # Update metrics file
        self._update_metrics_file("confusion_matrices", cm_data)
        
        print(f"Confusion matrix for round {round_num} logged.")
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(round_num, cm)
    
    def _update_metrics_file(self, metrics_type, metrics):
        """
        Update the metrics JSON file with new metrics.
        
        Args:
            metrics_type: Type of metrics (train, validation, test, client_metrics, confusion_matrices)
            metrics: The metrics to add
        """
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            if metrics_type == "client_metrics":
                client_id = list(metrics.keys())[0]
                if client_id not in data["client_metrics"]:
                    data["client_metrics"][client_id] = []
                data["client_metrics"][client_id].append(metrics[client_id])
            else:
                data[metrics_type].append(metrics)
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
        except Exception as e:
            print(f"Error updating metrics file: {e}")
    
    def plot_training_curves(self):
        """Plot and save training and validation curves."""
        if not self.train_metrics:
            print("No training metrics to plot.")
            return
        
        # Extract rounds and metrics
        rounds = [m["round"] for m in self.train_metrics]
        train_loss = [m["loss"] for m in self.train_metrics]
        train_accuracy = [m["accuracy"] for m in self.train_metrics]
        
        val_rounds = []
        val_loss = []
        val_accuracy = []
        
        if self.val_metrics:
            val_rounds = [m["round"] for m in self.val_metrics]
            val_loss = [m["loss"] for m in self.val_metrics]
            val_accuracy = [m["accuracy"] for m in self.val_metrics]
        
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, train_loss, 'b-', label='Training Loss')
        if val_loss:
            plt.plot(val_rounds, val_loss, 'r-', label='Validation Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "loss_curve.png"))
        plt.close()
        
        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, train_accuracy, 'b-', label='Training Accuracy')
        if val_accuracy:
            plt.plot(val_rounds, val_accuracy, 'r-', label='Validation Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "plots", "accuracy_curve.png"))
        plt.close()
        
        print("Training curves plotted and saved.")
    
    def plot_confusion_matrix(self, round_num, cm):
        """
        Plot and save confusion matrix.
        
        Args:
            round_num: The round number
            cm: Confusion matrix as a 2x2 list
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Round {round_num}')
        plt.colorbar()
        
        classes = ['Normal', 'Sybil']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = np.max(cm) / 2.
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                plt.text(j, i, format(cm[i][j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i][j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, "plots", f"confusion_matrix_round_{round_num}.png"))
        plt.close()
        
        print(f"Confusion matrix for round {round_num} plotted and saved.")
    
    def plot_roc_curve(self, y_true, y_pred_prob):
        """
        Plot and save ROC curve.
        
        Args:
            y_true: True labels
            y_pred_prob: Predicted probabilities
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, "plots", "roc_curve.png"))
        plt.close()
        
        print("ROC curve plotted and saved.")
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_prob):
        """
        Plot and save precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_prob: Predicted probabilities
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        avg_precision = average_precision_score(y_true, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, "plots", "precision_recall_curve.png"))
        plt.close()
        
        print("Precision-recall curve plotted and saved.")
        
        return avg_precision
    
    def generate_summary_report(self):
        """Generate and save a summary report of all metrics."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "training_summary": {
                "num_rounds": len(self.train_metrics),
                "final_loss": self.train_metrics[-1]["loss"] if self.train_metrics else None,
                "final_accuracy": self.train_metrics[-1]["accuracy"] if self.train_metrics else None,
                "best_accuracy": max([m["accuracy"] for m in self.train_metrics]) if self.train_metrics else None,
                "best_accuracy_round": [m["round"] for m in self.train_metrics if m["accuracy"] == max([m["accuracy"] for m in self.train_metrics])][0] if self.train_metrics else None
            },
            "validation_summary": {
                "num_validations": len(self.val_metrics),
                "final_loss": self.val_metrics[-1]["loss"] if self.val_metrics else None,
                "final_accuracy": self.val_metrics[-1]["accuracy"] if self.val_metrics else None,
                "best_accuracy": max([m["accuracy"] for m in self.val_metrics]) if self.val_metrics else None,
                "best_accuracy_round": [m["round"] for m in self.val_metrics if m["accuracy"] == max([m["accuracy"] for m in self.val_metrics])][0] if self.val_metrics else None
            },
            "test_summary": {
                "loss": self.test_metrics[-1]["loss"] if self.test_metrics else None,
                "accuracy": self.test_metrics[-1]["accuracy"] if self.test_metrics else None,
                "auc": self.test_metrics[-1].get("auc", None) if self.test_metrics else None
            },
            "confusion_matrix_summary": {
                "final_precision": self.confusion_matrices[-1]["precision"] if self.confusion_matrices else None,
                "final_recall": self.confusion_matrices[-1]["recall"] if self.confusion_matrices else None,
                "final_f1_score": self.confusion_matrices[-1]["f1_score"] if self.confusion_matrices else None
            },
            "client_participation": {
                "num_clients": len(self.client_metrics),
                "client_ids": list(self.client_metrics.keys())
            }
        }
        
        # Save report
        report_file = os.path.join(self.output_dir, "summary_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
        
        # Generate text report
        text_report = f"""
        ======================================================
        FEDERATED LEARNING EXPERIMENT SUMMARY
        ======================================================
        Timestamp: {report['timestamp']}
        
        TRAINING SUMMARY:
        - Number of rounds: {report['training_summary']['num_rounds']}
        - Final loss: {report['training_summary']['final_loss']:.4f if report['training_summary']['final_loss'] is not None else 'N/A'}
        - Final accuracy: {report['training_summary']['final_accuracy']:.4f if report['training_summary']['final_accuracy'] is not None else 'N/A'}
        - Best accuracy: {report['training_summary']['best_accuracy']:.4f if report['training_summary']['best_accuracy'] is not None else 'N/A'} (Round {report['training_summary']['best_accuracy_round']})
        
        VALIDATION SUMMARY:
        - Number of validations: {report['validation_summary']['num_validations']}
        - Final loss: {report['validation_summary']['final_loss']:.4f if report['validation_summary']['final_loss'] is not None else 'N/A'}
        - Final accuracy: {report['validation_summary']['final_accuracy']:.4f if report['validation_summary']['final_accuracy'] is not None else 'N/A'}
        - Best accuracy: {report['validation_summary']['best_accuracy']:.4f if report['validation_summary']['best_accuracy'] is not None else 'N/A'} (Round {report['validation_summary']['best_accuracy_round']})
        
        TEST SUMMARY:
        - Loss: {report['test_summary']['loss']:.4f if report['test_summary']['loss'] is not None else 'N/A'}
        - Accuracy: {report['test_summary']['accuracy']:.4f if report['test_summary']['accuracy'] is not None else 'N/A'}
        - AUC: {report['test_summary']['auc']:.4f if report['test_summary']['auc'] is not None else 'N/A'}
        
        CONFUSION MATRIX SUMMARY:
        - Final precision: {report['confusion_matrix_summary']['final_precision']:.4f if report['confusion_matrix_summary']['final_precision'] is not None else 'N/A'}
        - Final recall: {report['confusion_matrix_summary']['final_recall']:.4f if report['confusion_matrix_summary']['final_recall'] is not None else 'N/A'}
        - Final F1 score: {report['confusion_matrix_summary']['final_f1_score']:.4f if report['confusion_matrix_summary']['final_f1_score'] is not None else 'N/A'}
        
        CLIENT PARTICIPATION:
        - Number of clients: {report['client_participation']['num_clients']}
        - Client IDs: {', '.join(report['client_participation']['client_ids']) if report['client_participation']['client_ids'] else 'N/A'}
        
        ======================================================
        """
        
        text_report_file = os.path.join(self.output_dir, "summary_report.txt")
        with open(text_report_file, 'w') as f:
            f.write(text_report)
        
        print(f"Summary report generated and saved to {report_file} and {text_report_file}")
        
        return report
    
    def save_experiment_config(self, config):
        """
        Save experiment configuration.
        
        Args:
            config: Dictionary containing experiment configuration
        """
        config["timestamp"] = datetime.now().isoformat()
        
        # Update metrics file with config
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            data["experiment_config"] = config
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
        except Exception as e:
            print(f"Error updating metrics file with config: {e}")
        
        # Save config to separate file
        config_file = os.path.join(self.output_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"Experiment configuration saved to {config_file}")
