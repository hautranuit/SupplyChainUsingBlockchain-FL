"""
Data Processor for Federated Learning integration.
This module processes raw blockchain data into formats suitable for FL models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import tensorflow as tf

# JSON encoder tùy chỉnh để xử lý kiểu dữ liệu NumPy
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder hỗ trợ các kiểu dữ liệu NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../fl_integration_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_processor")

class DataProcessor:
    """
    Data Processor class for preparing blockchain data for FL models.
    Processes raw data into formats suitable for different FL models.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the data processor.
        
        Args:
            cache_dir: Directory to store processed data
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Data processor initialized with cache directory: {self.cache_dir}")
    
    def preprocess_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess real data from demo_context.json for federated learning models.
        
        Args:
            input_data: Data from demo_context.json containing nodes, products, batches, disputes
            
        Returns:
            Preprocessed data ready for FL training with format:
            {
                'sybil_detection': {'features': [...], 'labels': [...]},
                'batch_monitoring': {'features': [...], 'labels': [...]},
                'node_behavior': {'features': [...], 'labels': [...]}
            }
        """
        logger.info("Starting data preprocessing for FL models using real demo_context.json data")
        
        def safe_float(value, default=0.0):
            """Safely convert value to float, handling None values"""
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            """Safely convert value to int, handling None values"""
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        try:
            # Extract real data from demo_context.json structure
            nodes_data = input_data.get("nodes", {})
            products_data = input_data.get("products", {})
            batches_data = input_data.get("batches", {})
            disputes_data = input_data.get("disputes", {})
            
            logger.info(f"Real data summary: {len(nodes_data)} nodes, {len(products_data)} products, {len(batches_data)} batches, {len(disputes_data)} disputes")
            
            # Prepare data for each FL model
            processed_data = {
                'sybil_detection': {'features': [], 'labels': []},
                'batch_monitoring': {'features': [], 'labels': []},
                'node_behavior': {'features': [], 'labels': []}
            }
            
            # Process Sybil Detection data from real node interactions
            if nodes_data:
                logger.info("Extracting features for Sybil detection from real node data")
                for node_address, node_info in nodes_data.items():
                    interactions = node_info.get('interactions', [])
                    
                    # Count different types of interactions
                    interaction_types = {}
                    for interaction in interactions:
                        interaction_type = interaction.get('type', 'Unknown')
                        interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
                    
                    # Calculate suspicious behavior indicators
                    vote_actions = interaction_types.get('VoteBatch', 0) + interaction_types.get('VoteForArbitrator', 0)
                    transfer_actions = interaction_types.get('TransferNFT', 0) + interaction_types.get('TransferNFTViaBatch', 0)
                    receive_actions = interaction_types.get('ReceiveNFT', 0) + interaction_types.get('ReceiveNFTViaBatch', 0)
                    
                    # Features for sybil detection
                    features = [
                        safe_float(node_info.get('currentReputation', 0)) / 100.0,  # Normalized
                        safe_float(node_info.get('initialReputation', 0)) / 100.0,  # Normalized
                        1.0 if node_info.get('isVerified', False) else 0.0,
                        safe_float(len(interactions)) / 20.0,  # Normalized interaction count
                        safe_float(vote_actions) / 5.0,  # Normalized vote count
                        safe_float(transfer_actions + receive_actions) / 10.0,  # Normalized transfer activity
                    ]
                    
                    # Sybil detection: Mark nodes with suspicious patterns
                    # Very low reputation + high activity could indicate sybil
                    current_rep = node_info.get('currentReputation', 100)
                    is_sybil = 1.0 if (current_rep < 50 and len(interactions) > 5) else 0.0
                    
                    processed_data['sybil_detection']['features'].append(features)
                    processed_data['sybil_detection']['labels'].append(is_sybil)
            
            # Process Batch Monitoring data from real batch information
            if batches_data:
                logger.info("Extracting features for Batch monitoring from real batch data")
                for batch_id, batch_info in batches_data.items():
                    transactions = batch_info.get('transactions', [])
                    votes = batch_info.get('votes', {})
                    
                    # Calculate batch health metrics
                    total_votes = len(votes)
                    positive_votes = sum(1 for vote_info in votes.values() if vote_info.get('vote', False))
                    approval_rate = positive_votes / total_votes if total_votes > 0 else 0.0
                    
                    # Time analysis
                    propose_time = safe_int(batch_info.get('proposeTimestamp', 0))
                    commit_time = safe_int(batch_info.get('commitTimestamp', propose_time))
                    processing_duration = commit_time - propose_time if commit_time > propose_time else 0
                    
                    # Features for batch monitoring
                    features = [
                        safe_float(len(transactions)) / 5.0,  # Normalized transaction count
                        safe_float(total_votes) / 5.0,  # Normalized vote count
                        safe_float(approval_rate),  # Vote approval rate
                        safe_float(processing_duration) / 3600.0,  # Processing time in hours
                        1.0 if batch_info.get('status') == 'Committed' else 0.0,  # Commitment status
                        safe_float(len(batch_info.get('selectedValidators', []))) / 5.0,  # Validator count
                    ]
                    
                    # Anomaly detection: batches with low approval or long processing time
                    has_anomaly = 1.0 if (approval_rate < 0.6 or processing_duration > 1800) else 0.0
                    
                    processed_data['batch_monitoring']['features'].append(features)
                    processed_data['batch_monitoring']['labels'].append(has_anomaly)
            
            # Process Node Behavior data from interaction patterns
            if nodes_data:
                logger.info("Extracting features for Node behavior analysis from real interaction patterns")
                for node_address, node_info in nodes_data.items():
                    interactions = node_info.get('interactions', [])
                    if len(interactions) >= 2:  # Need at least 2 interactions for pattern analysis
                        
                        # Analyze interaction timing patterns
                        timestamps = [safe_int(inter.get('timestamp', 0)) for inter in interactions]
                        timestamps.sort()
                        
                        # Calculate timing metrics
                        time_intervals = []
                        for i in range(1, len(timestamps)):
                            interval = timestamps[i] - timestamps[i-1]
                            time_intervals.append(interval)
                        
                        avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0
                        min_interval = min(time_intervals) if time_intervals else 0
                        
                        # Count interaction types
                        interaction_types = set(inter.get('type', '') for inter in interactions)
                        dispute_actions = sum(1 for inter in interactions if 'Dispute' in inter.get('type', ''))
                        
                        # Features for behavior analysis
                        features = [
                            safe_float(len(interactions)) / 20.0,  # Normalized interaction count
                            safe_float(len(interaction_types)) / 10.0,  # Interaction diversity
                            safe_float(avg_interval) / 3600.0,  # Average time between actions (hours)
                            safe_float(min_interval) / 60.0,  # Minimum time between actions (minutes)
                            safe_float(dispute_actions) / 3.0,  # Normalized dispute involvement
                            safe_float(node_info.get('currentReputation', 0)) / 100.0,  # Reputation
                        ]
                        
                        # Suspicious behavior: too frequent actions or too many disputes
                        is_suspicious = 1.0 if (min_interval < 30 or dispute_actions > 1) else 0.0
                        
                        processed_data['node_behavior']['features'].append(features)
                        processed_data['node_behavior']['labels'].append(is_suspicious)
            
            # Process disputes for additional anomaly detection data
            if disputes_data:
                logger.info("Processing dispute data for additional insights")
                for dispute_id, dispute_info in disputes_data.items():
                    # Extract dispute features for node behavior analysis
                    disputer = dispute_info.get('disputer', '')
                    
                    # Add dispute-based features to relevant nodes
                    if disputer in nodes_data:
                        dispute_features = [
                            1.0,  # Has dispute flag
                            1.0 if dispute_info.get('status') == 'Enforced' else 0.0,  # Dispute enforced
                            safe_float(len(dispute_info.get('proposedCandidates', []))) / 5.0,  # Arbitrator candidates
                            safe_float(len(dispute_info.get('votes', {}))) / 5.0,  # Vote count in dispute
                            safe_float(dispute_info.get('resolutionOutcome', 0)),  # Resolution outcome
                            0.5  # Base reputation impact
                        ]
                        
                        # Mark as having dispute-related behavior
                        processed_data['node_behavior']['features'].append(dispute_features)
                        processed_data['node_behavior']['labels'].append(1.0)  # Dispute involvement is noteworthy
            
            # Ensure minimum data for training
            min_samples = 10
            for model_name, model_data in processed_data.items():
                current_samples = len(model_data['features'])
                logger.info(f"{model_name}: {current_samples} real samples extracted")
                
                if current_samples < min_samples:
                    logger.info(f"Augmenting {model_name} with synthetic data ({min_samples - current_samples} samples)")
                    
                    # Get feature dimension from existing data or set default
                    feature_dim = len(model_data['features'][0]) if model_data['features'] else 6
                    
                    # Generate synthetic data with realistic distributions
                    np.random.seed(42 + hash(model_name) % 1000)  # Different seed per model
                    num_synthetic = min_samples - current_samples
                    
                    for _ in range(num_synthetic):
                        # Generate features with realistic ranges based on the model type
                        if model_name == 'sybil_detection':
                            synthetic_features = [
                                np.random.beta(4, 2),  # Reputation tends to be higher
                                np.random.beta(4, 2),  # Initial reputation
                                np.random.choice([0.0, 1.0], p=[0.1, 0.9]),  # Most nodes verified
                                np.random.exponential(0.3),  # Interaction count
                                np.random.exponential(0.2),  # Vote actions
                                np.random.exponential(0.4),  # Transfer actions
                            ]
                        elif model_name == 'batch_monitoring':
                            synthetic_features = [
                                np.random.poisson(2) / 5.0,  # Transaction count
                                np.random.poisson(3) / 5.0,  # Vote count
                                np.random.beta(8, 2),  # Approval rate (tends high)
                                np.random.exponential(0.5),  # Processing duration
                                np.random.choice([0.0, 1.0], p=[0.2, 0.8]),  # Most batches committed
                                np.random.poisson(3) / 5.0,  # Validator count
                            ]
                        else:  # node_behavior
                            synthetic_features = [
                                np.random.exponential(0.4),  # Interaction count
                                np.random.exponential(0.3),  # Interaction diversity
                                np.random.exponential(2.0),  # Avg interval
                                np.random.exponential(0.5),  # Min interval
                                np.random.poisson(0.1) / 3.0,  # Dispute actions (rare)
                                np.random.beta(4, 2),  # Reputation
                            ]
                        
                        # Clip features to reasonable ranges
                        synthetic_features = [max(0.0, min(1.0, f)) for f in synthetic_features]
                        
                        # Generate labels with model-appropriate distributions
                        if model_name == 'sybil_detection':
                            synthetic_label = float(np.random.choice([0, 1], p=[0.9, 0.1]))  # 10% sybil
                        elif model_name == 'batch_monitoring':
                            synthetic_label = float(np.random.choice([0, 1], p=[0.85, 0.15]))  # 15% anomalies
                        else:  # node_behavior
                            synthetic_label = float(np.random.choice([0, 1], p=[0.8, 0.2]))  # 20% suspicious
                        
                        model_data['features'].append(synthetic_features)
                        model_data['labels'].append(synthetic_label)
            
            # Log final data summary
            for model_name, model_data in processed_data.items():
                logger.info(f"{model_name}: {len(model_data['features'])} samples, {sum(model_data['labels'])} positive labels")
            
            logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def process_node_data(self, 
                         node_data: Dict[str, Any], 
                         transaction_data: List[Dict[str, Any]] = None,
                         event_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process node data for FL models.
        
        Args:
            node_data: Raw node data
            transaction_data: Transaction data for the node
            event_data: Event data related to the node
            
        Returns:
            Processed node data
        """
        processed_data = {
            "address": node_data.get("address", ""),
            "features": {}
        }
        
        # Extract basic features
        processed_data["features"]["reputation"] = node_data.get("reputation", 0)
        
        # Process transaction data if available
        if transaction_data:
            # Calculate transaction frequency
            if len(transaction_data) > 0:
                # Get timestamps of transactions
                timestamps = []
                for tx in transaction_data:
                    if "timestamp" in tx:
                        timestamps.append(tx["timestamp"])
                
                if timestamps:
                    # Sort timestamps
                    timestamps.sort()
                    
                    # Calculate time differences between consecutive transactions
                    time_diffs = []
                    for i in range(1, len(timestamps)):
                        time_diffs.append(timestamps[i] - timestamps[i-1])
                    
                    # Calculate statistics
                    if time_diffs:
                        processed_data["features"]["tx_frequency_mean"] = np.mean(time_diffs)
                        processed_data["features"]["tx_frequency_std"] = np.std(time_diffs)
                        processed_data["features"]["tx_frequency_min"] = np.min(time_diffs)
                        processed_data["features"]["tx_frequency_max"] = np.max(time_diffs)
            
            # Calculate transaction value statistics
            values = []
            for tx in transaction_data:
                if "value" in tx:
                    values.append(float(tx["value"]))
            
            if values:
                processed_data["features"]["tx_value_mean"] = np.mean(values)
                processed_data["features"]["tx_value_std"] = np.std(values)
                processed_data["features"]["tx_value_min"] = np.min(values)
                processed_data["features"]["tx_value_max"] = np.max(values)
                processed_data["features"]["tx_value_total"] = np.sum(values)
        
        # Process event data if available
        if event_data:
            # Count events by type
            event_counts = {}
            for event in event_data:
                event_type = event.get("event", "unknown")
                if event_type not in event_counts:
                    event_counts[event_type] = 0
                event_counts[event_type] += 1
            
            # Add event counts to features
            for event_type, count in event_counts.items():
                processed_data["features"][f"event_{event_type}_count"] = count
        
        return processed_data
    
    def process_batch_data(self, 
                          batch_data: Dict[str, Any],
                          related_events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process batch data for FL models.
        
        Args:
            batch_data: Raw batch data
            related_events: Events related to the batch
            
        Returns:
            Processed batch data
        """
        processed_data = {
            "batch_id": batch_data.get("batchId", ""),
            "features": {}
        }
        
        # Extract basic features
        processed_data["features"]["validated"] = 1 if batch_data.get("validated", False) else 0
        processed_data["features"]["committed"] = 1 if batch_data.get("committed", False) else 0
        processed_data["features"]["approvals"] = int(batch_data.get("approvals", 0))
        processed_data["features"]["denials"] = int(batch_data.get("denials", 0))
        processed_data["features"]["flagged"] = 1 if batch_data.get("flagged", False) else 0
        
        # Calculate approval ratio
        total_votes = processed_data["features"]["approvals"] + processed_data["features"]["denials"]
        if total_votes > 0:
            processed_data["features"]["approval_ratio"] = processed_data["features"]["approvals"] / total_votes
        else:
            processed_data["features"]["approval_ratio"] = 0
        
        # Process related events if available
        if related_events:
            # Count events by type
            event_counts = {}
            for event in related_events:
                event_type = event.get("event", "unknown")
                if event_type not in event_counts:
                    event_counts[event_type] = 0
                event_counts[event_type] += 1
            
            # Add event counts to features
            for event_type, count in event_counts.items():
                processed_data["features"][f"event_{event_type}_count"] = count
        
        return processed_data
    
    def process_dispute_data(self, 
                            dispute_data: Dict[str, Any],
                            related_events: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process dispute data for FL models.
        
        Args:
            dispute_data: Raw dispute data
            related_events: Events related to the dispute
            
        Returns:
            Processed dispute data
        """
        processed_data = {
            "dispute_id": dispute_data.get("disputeId", ""),
            "features": {}
        }
        
        # Extract basic features
        processed_data["features"]["resolved"] = 1 if dispute_data.get("resolved", False) else 0
        processed_data["features"]["upheld"] = 1 if dispute_data.get("upheld", False) else 0
        processed_data["features"]["votes_for"] = int(dispute_data.get("votesFor", 0))
        processed_data["features"]["votes_against"] = int(dispute_data.get("votesAgainst", 0))
        
        # Calculate vote ratio
        total_votes = processed_data["features"]["votes_for"] + processed_data["features"]["votes_against"]
        if total_votes > 0:
            processed_data["features"]["vote_ratio"] = processed_data["features"]["votes_for"] / total_votes
        else:
            processed_data["features"]["vote_ratio"] = 0
        
        # Process related events if available
        if related_events:
            # Count events by type
            event_counts = {}
            for event in related_events:
                event_type = event.get("event", "unknown")
                if event_type not in event_counts:
                    event_counts[event_type] = 0
                event_counts[event_type] += 1
            
            # Add event counts to features
            for event_type, count in event_counts.items():
                processed_data["features"][f"event_{event_type}_count"] = count
        
        return processed_data
    
    def prepare_sybil_detection_data(self, 
                                    nodes_data: Dict[str, Any],
                                    transactions_data: Dict[str, List[Dict[str, Any]]],
                                    events_data: Dict[str, List[Dict[str, Any]]],
                                    sybil_nodes: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Sybil detection model.
        
        Args:
            nodes_data: Dictionary of node data
            transactions_data: Dictionary of transaction data by node
            events_data: Dictionary of event data by node
            sybil_nodes: List of known Sybil node addresses
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        # Process each node
        for node_address, node_data in nodes_data.items():
            # Get transactions and events for this node
            node_transactions = transactions_data.get(node_address, [])
            node_events = events_data.get(node_address, [])
            
            # Process node data
            processed_node = self.process_node_data(
                node_data,
                node_transactions,
                node_events
            )
            
            # Extract features
            node_features = []
            
            # Add reputation
            node_features.append(processed_node["features"].get("reputation", 0))
            
            # Add transaction frequency features
            node_features.append(processed_node["features"].get("tx_frequency_mean", 0))
            node_features.append(processed_node["features"].get("tx_frequency_std", 0))
            
            # Add transaction value features
            node_features.append(processed_node["features"].get("tx_value_mean", 0))
            node_features.append(processed_node["features"].get("tx_value_std", 0))
            node_features.append(processed_node["features"].get("tx_value_total", 0))
            
            # Add event counts
            node_features.append(processed_node["features"].get("event_BatchValidated_count", 0))
            node_features.append(processed_node["features"].get("event_BatchProposed_count", 0))
            node_features.append(processed_node["features"].get("event_DisputeCreated_count", 0))
            node_features.append(processed_node["features"].get("event_DisputeResolved_count", 0))
            
            # Add features to list
            features.append(node_features)
            
            # Determine label (1 for Sybil, 0 for normal)
            is_sybil = 0
            if sybil_nodes and node_address in sybil_nodes:
                is_sybil = 1
            
            labels.append(is_sybil)
        
        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Normalize features
        features_mean = np.mean(features_array, axis=0)
        features_std = np.std(features_array, axis=0)
        features_std[features_std == 0] = 1  # Avoid division by zero
        features_array = (features_array - features_mean) / features_std
        
        return features_array, labels_array
    
    def prepare_batch_monitoring_data(self, 
                                     batches_data: Dict[str, Any],
                                     events_data: Dict[str, List[Dict[str, Any]]],
                                     flagged_batches: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for batch monitoring model.
        
        Args:
            batches_data: Dictionary of batch data
            events_data: Dictionary of event data by batch
            flagged_batches: List of known flagged batch IDs
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        # Process each batch
        for batch_id, batch_data in batches_data.items():
            # Get events for this batch
            batch_events = events_data.get(batch_id, [])
            
            # Process batch data
            processed_batch = self.process_batch_data(
                batch_data,
                batch_events
            )
            
            # Extract features
            batch_features = []
            
            # Add basic features
            batch_features.append(processed_batch["features"].get("approvals", 0))
            batch_features.append(processed_batch["features"].get("denials", 0))
            batch_features.append(processed_batch["features"].get("approval_ratio", 0))
            
            # Add event counts
            batch_features.append(processed_batch["features"].get("event_BatchValidated_count", 0))
            batch_features.append(processed_batch["features"].get("event_BatchProposed_count", 0))
            
            # Add features to list
            features.append(batch_features)
            
            # Determine label (1 for flagged, 0 for normal)
            is_flagged = 0
            if flagged_batches and batch_id in flagged_batches:
                is_flagged = 1
            elif processed_batch["features"].get("flagged", 0) == 1:
                is_flagged = 1
            
            labels.append(is_flagged)
        
        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Normalize features
        if len(features_array) > 0:
            features_mean = np.mean(features_array, axis=0)
            features_std = np.std(features_array, axis=0)
            features_std[features_std == 0] = 1  # Avoid division by zero
            features_array = (features_array - features_mean) / features_std
        
        return features_array, labels_array
    
    def prepare_node_behavior_data(self, 
                                  nodes_data: Dict[str, Any],
                                  transactions_data: Dict[str, List[Dict[str, Any]]],
                                  events_data: Dict[str, List[Dict[str, Any]]],
                                  time_window: int = 10,
                                  anomalous_nodes: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for node behavior model.
        
        Args:
            nodes_data: Dictionary of node data
            transactions_data: Dictionary of transaction data by node
            events_data: Dictionary of event data by node
            time_window: Number of time steps in the time series
            anomalous_nodes: List of known anomalous node addresses
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        # Process each node
        for node_address, node_data in nodes_data.items():
            # Get transactions and events for this node
            node_transactions = transactions_data.get(node_address, [])
            node_events = events_data.get(node_address, [])
            
            # Sort transactions by timestamp
            if node_transactions and "timestamp" in node_transactions[0]:
                node_transactions.sort(key=lambda x: x["timestamp"])
            
            # Sort events by timestamp
            if node_events and "timestamp" in node_events[0]:
                node_events.sort(key=lambda x: x["timestamp"])
            
            # Create time series data
            time_series = []
            
            # If we have enough data, create time series
            if len(node_transactions) >= time_window:
                # Create time series for each feature
                for i in range(len(node_transactions) - time_window + 1):
                    window_transactions = node_transactions[i:i+time_window]
                    
                    # Extract features for this window
                    window_features = []
                    
                    # Transaction values
                    values = [float(tx.get("value", 0)) for tx in window_transactions]
                    window_features.append(np.mean(values))
                    window_features.append(np.std(values))
                    window_features.append(np.max(values))
                    
                    # Transaction frequencies
                    if "timestamp" in window_transactions[0]:
                        timestamps = [tx["timestamp"] for tx in window_transactions]
                        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                        if time_diffs:
                            window_features.append(np.mean(time_diffs))
                            window_features.append(np.std(time_diffs))
                        else:
                            window_features.append(0)
                            window_features.append(0)
                    else:
                        window_features.append(0)
                        window_features.append(0)
                    
                    # Count events in this time window
                    if node_events and "timestamp" in node_events[0]:
                        start_time = window_transactions[0]["timestamp"]
                        end_time = window_transactions[-1]["timestamp"]
                        
                        window_events = [e for e in node_events if start_time <= e["timestamp"] <= end_time]
                        
                        # Count events by type
                        event_counts = {}
                        for event in window_events:
                            event_type = event.get("event", "unknown")
                            if event_type not in event_counts:
                                event_counts[event_type] = 0
                            event_counts[event_type] += 1
                        
                        # Add event counts
                        window_features.append(event_counts.get("BatchValidated", 0))
                        window_features.append(event_counts.get("BatchProposed", 0))
                        window_features.append(event_counts.get("DisputeCreated", 0))
                        window_features.append(event_counts.get("DisputeResolved", 0))
                    else:
                        window_features.extend([0, 0, 0, 0])
                    
                    # Add window features to time series
                    time_series.append(window_features)
            
            # If we have time series data, add to features
            if time_series:
                features.append(time_series)
                
                # Determine label (1 for anomalous, 0 for normal)
                is_anomalous = 0
                if anomalous_nodes and node_address in anomalous_nodes:
                    is_anomalous = 1
                
                labels.append(is_anomalous)
        
        # Convert to numpy arrays
        if features:
            features_array = np.array(features, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)
            
            # Normalize features
            features_mean = np.mean(features_array, axis=(0, 1))
            features_std = np.std(features_array, axis=(0, 1))
            features_std[features_std == 0] = 1  # Avoid division by zero
            features_array = (features_array - features_mean) / features_std
            
            return features_array, labels_array
        else:
            # Return empty arrays if no features
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    
    def prepare_dispute_risk_data(self, 
                                 disputes_data: Dict[str, Any],
                                 events_data: Dict[str, List[Dict[str, Any]]],
                                 high_risk_disputes: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for dispute risk model.
        
        Args:
            disputes_data: Dictionary of dispute data
            events_data: Dictionary of event data by dispute
            high_risk_disputes: List of known high-risk dispute IDs
            
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        
        # Process each dispute
        for dispute_id, dispute_data in disputes_data.items():
            # Get events for this dispute
            dispute_events = events_data.get(dispute_id, [])
            
            # Process dispute data
            processed_dispute = self.process_dispute_data(
                dispute_data,
                dispute_events
            )
            
            # Extract features
            dispute_features = []
            
            # Add basic features
            dispute_features.append(processed_dispute["features"].get("votes_for", 0))
            dispute_features.append(processed_dispute["features"].get("votes_against", 0))
            dispute_features.append(processed_dispute["features"].get("vote_ratio", 0))
            
            # Add event counts
            dispute_features.append(processed_dispute["features"].get("event_DisputeCreated_count", 0))
            dispute_features.append(processed_dispute["features"].get("event_DisputeResolved_count", 0))
            
            # Add features to list
            features.append(dispute_features)
            
            # Determine label (1 for high risk, 0 for low risk)
            is_high_risk = 0
            if high_risk_disputes and dispute_id in high_risk_disputes:
                is_high_risk = 1
            
            labels.append(is_high_risk)
        
        # Convert to numpy arrays
        features_array = np.array(features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32)
        
        # Normalize features
        if len(features_array) > 0:
            features_mean = np.mean(features_array, axis=0)
            features_std = np.std(features_array, axis=0)
            features_std[features_std == 0] = 1  # Avoid division by zero
            features_array = (features_array - features_mean) / features_std
        
        return features_array, labels_array
    
    def create_tf_dataset(self, 
                         features: np.ndarray, 
                         labels: np.ndarray, 
                         batch_size: int = 32,
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from features and labels.
        
        Args:
            features: Feature array
            labels: Label array
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            
        Returns:
            TensorFlow dataset
        """
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        return dataset
    
    def split_data(self, 
                  features: np.ndarray, 
                  labels: np.ndarray,
                  test_size: float = 0.2,
                  random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion of data to use for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_features, test_features, train_labels, test_labels)
        """
        # Set random seed
        np.random.seed(random_seed)
        
        # Get indices
        indices = np.random.permutation(len(features))
        test_count = int(test_size * len(features))
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        # Split data
        train_features = features[train_indices]
        test_features = features[test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        
        return train_features, test_features, train_labels, test_labels
    
    def create_federated_datasets(self, 
                                 features: np.ndarray, 
                                 labels: np.ndarray,
                                 num_clients: int = 3,
                                 batch_size: int = 32,
                                 random_seed: int = 42) -> List[tf.data.Dataset]:
        """
        Create federated datasets for FL training.
        
        Args:
            features: Feature array
            labels: Label array
            num_clients: Number of FL clients
            batch_size: Batch size for the datasets
            random_seed: Random seed for reproducibility
            
        Returns:
            List of TensorFlow datasets, one per client
        """
        # Set random seed
        np.random.seed(random_seed)
        
        # Get indices
        indices = np.random.permutation(len(features))
        
        # Split indices among clients
        client_indices = np.array_split(indices, num_clients)
        
        # Create datasets
        federated_datasets = []
        for client_idx in client_indices:
            client_features = features[client_idx]
            client_labels = labels[client_idx]
            
            # Create dataset
            client_dataset = self.create_tf_dataset(
                client_features,
                client_labels,
                batch_size=batch_size,
                shuffle=True
            )
            
            federated_datasets.append(client_dataset)
        
        return federated_datasets
    
    def save_processed_data(self, 
                           data: Any, 
                           filename: str):
        """
        Save processed data to a file.
        
        Args:
            data: Data to save
            filename: Name of the file
        """
        file_path = os.path.join(self.cache_dir, filename)
        
        try:
            # Save data based on type
            if isinstance(data, np.ndarray):
                np.save(file_path, data)
                logger.info(f"Saved numpy array to {file_path}.npy")
            elif isinstance(data, (dict, list)):
                with open(f"{file_path}.json", 'w') as f:
                    json.dump(data, f, indent=2, cls=NumpyJSONEncoder)
                logger.info(f"Saved JSON data to {file_path}.json")
            else:
                logger.error(f"Unsupported data type for saving: {type(data)}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
    
    def load_processed_data(self, 
                           filename: str, 
                           data_type: str = "auto") -> Any:
        """
        Load processed data from a file.
        
        Args:
            filename: Name of the file
            data_type: Type of data to load ("numpy", "json", or "auto")
            
        Returns:
            Loaded data
        """
        # Determine file path based on data type
        if data_type == "auto":
            # Try numpy first
            numpy_path = os.path.join(self.cache_dir, f"{filename}.npy")
            json_path = os.path.join(self.cache_dir, f"{filename}.json")
            
            if os.path.exists(numpy_path):
                file_path = numpy_path
                data_type = "numpy"
            elif os.path.exists(json_path):
                file_path = json_path
                data_type = "json"
            else:
                # Try without extension
                base_path = os.path.join(self.cache_dir, filename)
                if os.path.exists(base_path):
                    file_path = base_path
                    # Guess type based on content
                    with open(file_path, 'rb') as f:
                        if f.read(6) == b'\x93NUMPY':
                            data_type = "numpy"
                        else:
                            data_type = "json"
                else:
                    logger.error(f"File not found: {filename}")
                    return None
        else:
            # Use specified data type
            if data_type == "numpy":
                file_path = os.path.join(self.cache_dir, f"{filename}.npy")
            elif data_type == "json":
                file_path = os.path.join(self.cache_dir, f"{filename}.json")
            else:
                logger.error(f"Unsupported data type: {data_type}")
                return None
        
        try:
            # Load data based on type
            if data_type == "numpy":
                data = np.load(file_path)
                logger.info(f"Loaded numpy array from {file_path}")
                return data
            elif data_type == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded JSON data from {file_path}")
                return data
            else:
                logger.error(f"Unsupported data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            return None
    
    def process_blockchain_data(self, blockchain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw blockchain data into a format suitable for FL models.
        
        Args:
            blockchain_data: Raw blockchain data dictionary from connector
            
        Returns:
            Processed data with extracted features
        """
        logger.info(f"Processing blockchain data")
        
        processed_data = {
            "nodes": {},
            "batches": {},
            "disputes": {},
            "features": []
        }
        
        # Xử lý nodes từ blockchain_data
        for node_address, node_data in blockchain_data.get("nodes", {}).items():
            record_type = node_data.get("type", "node")
            
            if record_type == "node":
                # Process node data
                node_address = node_data.get("address", "")
                transactions = node_data.get("transactions", [])
                events = node_data.get("events", [])
                
                node_processed = self.process_node_data(node_data, transactions, events)
                processed_data["nodes"][node_address] = node_processed
                
                # Extract features for ML models
                if "features" in node_processed:
                    feature_dict = node_processed["features"]
                    # Convert dict to list for ML
                    feature_list = [feature_dict.get(key, 0) for key in sorted(feature_dict.keys())]
                    processed_data["features"].append(feature_list)
            
        
        # Xử lý batches từ blockchain_data
        for batch_id, batch_data in blockchain_data.get("batches", {}).items():
            record_type = batch_data.get("type", "batch")
            
            if record_type == "batch":
                # Process batch data
                related_events = batch_data.get("events", [])
                
                batch_processed = self.process_batch_data(batch_data, related_events)
                processed_data["batches"][batch_id] = batch_processed
                
                # Extract features for ML models
                if "features" in batch_processed:
                    feature_dict = batch_processed["features"]
                    # Convert dict to list for ML
                    feature_list = [feature_dict.get(key, 0) for key in sorted(feature_dict.keys())]
                    processed_data["features"].append(feature_list)
            
        
        # Xử lý disputes từ blockchain_data
        for dispute_id, dispute_data in blockchain_data.get("disputes", {}).items():
            record_type = dispute_data.get("type", "dispute")
            
            if record_type == "dispute":
                # Process dispute data
                related_events = dispute_data.get("events", [])
                
                dispute_processed = self.process_dispute_data(dispute_data, related_events)
                processed_data["disputes"][dispute_id] = dispute_processed
                
                # Extract features for ML models
                if "features" in dispute_processed:
                    feature_dict = dispute_processed["features"]
                    # Convert dict to list for ML
                    feature_list = [feature_dict.get(key, 0) for key in sorted(feature_dict.keys())]
                    processed_data["features"].append(feature_list)
        
        logger.info(f"Processed {len(processed_data['nodes'])} nodes, {len(processed_data['batches'])} batches, {len(processed_data['disputes'])} disputes")
        return processed_data
    
    def extract_sybil_detection_features(self, processed_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract features specifically for Sybil detection model.
        
        Args:
            processed_data: Processed blockchain data
            
        Returns:
            Dictionary mapping node IDs to feature vectors
        """
        sybil_features = {}
        
        for node_id, node_data in processed_data.get("nodes", {}).items():
            if "features" in node_data:
                feature_dict = node_data["features"]
                
                # Extract relevant features for Sybil detection
                relevant_features = []
                
                # Transaction patterns
                for key in ["tx_frequency_mean", "tx_frequency_std", "tx_value_mean", "tx_value_std"]:
                    relevant_features.append(feature_dict.get(key, 0))
                
                # Reputation and behavior patterns
                relevant_features.append(feature_dict.get("reputation", 0))
                
                # Event patterns
                for key in feature_dict.keys():
                    if key.startswith("event_") and key.endswith("_count"):
                        relevant_features.append(feature_dict.get(key, 0))
                
                sybil_features[node_id] = relevant_features
        
        logger.info(f"Extracted Sybil detection features for {len(sybil_features)} nodes")
        return sybil_features
    
    def extract_batch_monitoring_features(self, processed_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract features specifically for batch monitoring model.
        
        Args:
            processed_data: Processed blockchain data
            
        Returns:
            Dictionary mapping batch IDs to feature vectors
        """
        batch_features = {}
        
        for batch_id, batch_data in processed_data.get("batches", {}).items():
            if "features" in batch_data:
                feature_dict = batch_data["features"]
                
                # Extract relevant features for batch monitoring
                relevant_features = []
                
                # Approval patterns
                for key in ["approvals", "denials", "approval_ratio"]:
                    relevant_features.append(feature_dict.get(key, 0))
                
                # Status flags
                for key in ["validated", "committed", "flagged"]:
                    relevant_features.append(feature_dict.get(key, 0))
                
                # Event patterns
                for key in feature_dict.keys():
                    if key.startswith("event_") and key.endswith("_count"):
                        relevant_features.append(feature_dict.get(key, 0))
                
                batch_features[batch_id] = relevant_features
        
        logger.info(f"Extracted batch monitoring features for {len(batch_features)} batches")
        return batch_features
    
    def extract_node_behavior_features(self, processed_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract features specifically for node behavior analysis model.
        
        Args:
            processed_data: Processed blockchain data
            
        Returns:
            Dictionary mapping node IDs to feature vectors
        """
        node_features = {}
        
        # Process node data
        for node_id, node_data in processed_data.get("nodes", {}).items():
            if "features" in node_data:
                feature_dict = node_data["features"]
                
                # Extract relevant features for node behavior analysis
                relevant_features = []
                
                # Transaction patterns
                for key in ["tx_frequency_mean", "tx_frequency_std", "tx_value_mean", 
                           "tx_value_std", "tx_value_min", "tx_value_max", "tx_value_total"]:
                    relevant_features.append(feature_dict.get(key, 0))
                
                # Reputation
                relevant_features.append(feature_dict.get("reputation", 0))
                
                # Add batch validation statistics if available
                batch_approvals = 0
                batch_denials = 0
                
                # Count batches approved/denied by this node (simulated here)
                for batch_data in processed_data.get("batches", {}).values():
                    # In a real system, you'd have a relationship between nodes and batches
                    # For this implementation, we're using placeholder data
                    pass
                
                relevant_features.append(batch_approvals)
                relevant_features.append(batch_denials)
                
                # Event patterns
                for key in feature_dict.keys():
                    if key.startswith("event_") and key.endswith("_count"):
                        relevant_features.append(feature_dict.get(key, 0))
                
                node_features[node_id] = relevant_features
        
        logger.info(f"Extracted node behavior features for {len(node_features)} nodes")
        return node_features
