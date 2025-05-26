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
import traceback

# Optional TensorFlow import for FL dataset creation
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

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
        Enhanced preprocessing for real data from demo_context.json for federated learning models.
        Now includes comprehensive attack detection capabilities with ground truth labels.
        Processes both normal operational data and attack data for robust ML training.
        """
        logger.info("Starting enhanced data preprocessing with attack detection capabilities")

        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        # Enhanced blockchain interaction types mapping with attack patterns
        BLOCKCHAIN_INTERACTION_TYPES = {
            'MintProduct': 'production',
            'ListProduct': 'marketplace',
            'TransferNFT': 'transfer',
            'ReceiveNFT': 'transfer',
            'VoteBatch': 'governance',
            'VoteForArbitrator': 'governance',
            'StartTransport': 'logistics',
            'CompleteTransport': 'logistics',
            'FinalizePurchase': 'marketplace',
            'TransferNFTViaBatch': 'batch_transfer',
            'ReceiveNFTViaBatch': 'batch_transfer',
            'CommitBatchAttempt': 'batch_operations',
            'ProposeBatch': 'batch_operations',
            'MakeDisputeDecision': 'dispute',
            'InitiateDispute': 'dispute',
            'InitiatePurchase': 'marketplace',
            'DepositCollateral': 'marketplace',
            'OpenDispute': 'dispute',
            'ProposeArbitratorCandidate': 'governance',
            'SelectArbitrator': 'governance',
            # Attack-specific interaction types
            'SybilRegistration': 'attack_sybil',
            'CoordinatedVoting': 'attack_sybil',
            'ArtificialActivity': 'attack_sybil',
            'BribeReceived': 'attack_bribery',
            'CompromisedVoting': 'attack_bribery',
            'simulatedBehaviorChange': 'attack_bribery'
        }

        def is_valid_interaction(inter):
            """Enhanced validation that includes attack interaction patterns"""
            if not isinstance(inter, dict):
                return False
            if 'type' not in inter or 'timestamp' not in inter:
                return False
            if not isinstance(inter['type'], str) or not isinstance(inter['timestamp'], int):
                return False
            # Accept both normal and attack interaction types
            interaction_type = inter.get('type', '')
            if interaction_type not in BLOCKCHAIN_INTERACTION_TYPES:
                logger.warning(f"Unknown interaction type: {interaction_type}")
                return False
            return True

        def extract_attack_features(node_address: str, node_info: dict, attack_metadata: dict) -> dict:
            """Extract attack-specific features for enhanced detection"""
            attack_features = {
                'is_sybil': 0.0,
                'is_bribed': 0.0,
                'sybil_confidence': 0.0,
                'bribery_confidence': 0.0,
                'suspicious_pattern_count': 0.0,
                'attack_interaction_ratio': 0.0
            }
            
            # Check if node is marked as Sybil
            if node_address in attack_metadata.get('sybilNodes', []):
                attack_features['is_sybil'] = 1.0
                # Find Sybil indicators
                for indicator in attack_metadata.get('attackFeatures', {}).get('sybilIndicators', []):
                    if indicator.get('nodeAddress') == node_address:
                        attack_features['sybil_confidence'] = indicator.get('confidence', 0.0)
                        break
            
            # Check if node is marked as bribed
            if node_address in attack_metadata.get('bribedNodes', []):
                attack_features['is_bribed'] = 1.0
                # Find bribery indicators
                for indicator in attack_metadata.get('attackFeatures', {}).get('briberyIndicators', []):
                    if indicator.get('nodeAddress') == node_address:
                        attack_features['bribery_confidence'] = indicator.get('confidence', 0.0)
                        break
            
            # Count suspicious patterns
            suspicious_patterns = node_info.get('suspiciousPatterns', [])
            attack_features['suspicious_pattern_count'] = safe_float(len(suspicious_patterns)) / 10.0
            
            # Calculate attack interaction ratio
            interactions = node_info.get('interactions', [])
            attack_interactions = [i for i in interactions if 
                                 BLOCKCHAIN_INTERACTION_TYPES.get(i.get('type', ''), '').startswith('attack_')]
            total_interactions = len(interactions)
            if total_interactions > 0:
                attack_features['attack_interaction_ratio'] = len(attack_interactions) / total_interactions
            
            return attack_features

        try:
            nodes_data = input_data.get("nodes", {})
            products_data = input_data.get("products", {})
            batches_data = input_data.get("batches", {})
            disputes_data = input_data.get("disputes", {})
            attack_metadata = input_data.get("attackMetadata", {})
            ground_truth_labels = attack_metadata.get("groundTruthLabels", {})

            logger.info(f"Enhanced data summary: {len(nodes_data)} nodes, {len(products_data)} products, "
                       f"{len(batches_data)} batches, {len(disputes_data)} disputes")
            
            if attack_metadata:
                logger.info(f"Attack data detected: {len(attack_metadata.get('sybilNodes', []))} Sybil nodes, "
                           f"{len(attack_metadata.get('bribedNodes', []))} bribed nodes")

            processed_data = {
                'sybil_detection': {'features': [], 'labels': []},
                'bribery_detection': {'features': [], 'labels': []}
            }
            dropped_interactions = 0
            dropped_nodes = 0
            
            # --- Enhanced Sybil Detection with Attack Data ---
            if nodes_data:
                logger.info("Extracting enhanced features for Sybil detection with attack patterns")
                for node_address, node_info in nodes_data.items():
                    interactions = node_info.get('interactions', [])
                    valid_interactions = [i for i in interactions if is_valid_interaction(i)]
                    if len(valid_interactions) < 2:
                        dropped_nodes += 1
                        continue
                    
                    # Enhanced blockchain interaction categorization
                    interaction_categories = {
                        'production': 0, 'marketplace': 0, 'transfer': 0, 'governance': 0,
                        'logistics': 0, 'batch_operations': 0, 'dispute': 0,
                        'attack_sybil': 0, 'attack_bribery': 0
                    }
                    
                    for interaction in valid_interactions:
                        interaction_type = interaction.get('type', 'Unknown')
                        category = BLOCKCHAIN_INTERACTION_TYPES.get(interaction_type, 'unknown')
                        if category in interaction_categories:
                            interaction_categories[category] += 1
                    
                    # Calculate comprehensive features including attack indicators
                    vote_actions = interaction_categories['governance']
                    transfer_actions = interaction_categories['transfer'] + interaction_categories['batch_operations']
                    production_actions = interaction_categories['production']
                    marketplace_actions = interaction_categories['marketplace']
                    logistics_actions = interaction_categories['logistics']
                    dispute_actions = interaction_categories['dispute']
                    attack_sybil_actions = interaction_categories['attack_sybil']
                    attack_bribery_actions = interaction_categories['attack_bribery']
                    
                    # Extract attack-specific features
                    attack_features = extract_attack_features(node_address, node_info, attack_metadata)
                    
                    # Enhanced feature vector with attack detection capabilities
                    features = [
                        safe_float(node_info.get('currentReputation', 0)) / 100.0,
                        safe_float(node_info.get('initialReputation', 0)) / 100.0,
                        1.0 if node_info.get('isVerified', False) else 0.0,
                        safe_float(len(valid_interactions)) / 20.0,
                        safe_float(vote_actions) / 5.0,
                        safe_float(transfer_actions) / 10.0,
                        safe_float(production_actions) / 5.0,
                        safe_float(marketplace_actions) / 5.0,
                        safe_float(logistics_actions) / 5.0,
                        safe_float(dispute_actions) / 3.0,
                        # New attack detection features
                        safe_float(attack_sybil_actions) / 5.0,
                        attack_features['suspicious_pattern_count'],
                        attack_features['attack_interaction_ratio'],
                        attack_features['sybil_confidence'],
                    ]
                    
                    # Enhanced Sybil detection with ground truth labels
                    if node_address in ground_truth_labels.get('sybilDetection', {}):
                        is_sybil = ground_truth_labels['sybilDetection'][node_address]
                    else:
                        # Fallback to heuristic detection for nodes without ground truth
                        current_rep = node_info.get('currentReputation', 100)
                        suspicious_patterns = 0
                        if current_rep < 50: suspicious_patterns += 1
                        if len(valid_interactions) > 15: suspicious_patterns += 1
                        if dispute_actions > 2: suspicious_patterns += 1
                        if vote_actions > 10: suspicious_patterns += 1
                        if transfer_actions > 20: suspicious_patterns += 1
                        if attack_sybil_actions > 0: suspicious_patterns += 2  # Strong indicator
                        if node_info.get('isSybil', False): suspicious_patterns += 3  # Explicit marking
                        
                        is_sybil = 1.0 if suspicious_patterns >= 2 else 0.0
                    
                    processed_data['sybil_detection']['features'].append(features)
                    processed_data['sybil_detection']['labels'].append(is_sybil)
            
            # --- Enhanced Batch Monitoring with Attack Patterns ---
            if batches_data:
                logger.info("Extracting enhanced features for Batch monitoring with attack detection")
                for batch_id, batch_info in batches_data.items():
                    transactions = batch_info.get('transactions', [])
                    votes = batch_info.get('votes', {})
                    total_votes = len(votes)
                    positive_votes = sum(1 for vote_info in votes.values() if vote_info.get('vote', False))
                    approval_rate = positive_votes / total_votes if total_votes > 0 else 0.0
                    
                    # Check for attack-related patterns in batch
                    sybil_votes = 0
                    bribed_votes = 0
                    for voter_addr, vote_info in votes.items():
                        if voter_addr in attack_metadata.get('sybilNodes', []):
                            sybil_votes += 1
                        if voter_addr in attack_metadata.get('bribedNodes', []):
                            bribed_votes += 1
                    
                    propose_time = safe_int(batch_info.get('proposeTimestamp', 0))
                    commit_time = safe_int(batch_info.get('commitTimestamp', propose_time))
                    processing_duration = commit_time - propose_time if commit_time > propose_time else 0
                    
                    # Enhanced features with attack detection
                    features = [
                        safe_float(len(transactions)) / 5.0,
                        safe_float(total_votes) / 5.0,
                        safe_float(approval_rate),
                        safe_float(processing_duration) / 3600.0,
                        1.0 if batch_info.get('status') == 'Committed' else 0.0,
                        safe_float(len(batch_info.get('selectedValidators', []))) / 5.0,
                        # Attack detection features
                        safe_float(sybil_votes) / max(1, total_votes),  # Ratio of Sybil votes
                        safe_float(bribed_votes) / max(1, total_votes), # Ratio of bribed votes
                    ]
                    
                    # Enhanced anomaly detection
                    has_anomaly = 1.0 if (approval_rate < 0.6 or processing_duration > 1800 or 
                                        sybil_votes > 0 or bribed_votes > 0) else 0.0
                    
                    processed_data['batch_monitoring']['features'].append(features)
                    processed_data['batch_monitoring']['labels'].append(has_anomaly)
            
            # --- Enhanced Bribery Detection with Attack Detection ---
            if nodes_data:
                logger.info("Extracting enhanced features for Bribery detection with attack patterns")
                for node_address, node_info in nodes_data.items():
                    interactions = node_info.get('interactions', [])
                    valid_interactions = [i for i in interactions if is_valid_interaction(i)]
                    if len(valid_interactions) < 2:
                        dropped_nodes += 1
                        continue
                    
                    # Enhanced temporal analysis
                    timestamps = [safe_int(inter.get('timestamp', 0)) for inter in valid_interactions]
                    timestamps.sort()
                    time_intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0
                    min_interval = min(time_intervals) if time_intervals else 0
                    
                    # Enhanced behavioral metrics
                    interaction_diversity = len(set(inter.get('type', '') for inter in valid_interactions))
                    attack_features = extract_attack_features(node_address, node_info, attack_metadata)
                    
                    # Calculate behavioral change indicators
                    behavioral_change_score = 0.0
                    if node_info.get('isBribed', False):
                        behavioral_change_score = 1.0
                    elif attack_features['attack_interaction_ratio'] > 0.1:
                        behavioral_change_score = 0.8
                    elif len(node_info.get('suspiciousPatterns', [])) > 0:
                        behavioral_change_score = 0.6
                    
                    # Enhanced features for bribery detection (15 features)
                    features = [
                        safe_float(len(valid_interactions)) / 20.0,
                        safe_float(interaction_diversity) / 15.0,
                        safe_float(avg_interval) / 3600.0,
                        safe_float(min_interval) / 60.0,
                        safe_float(node_info.get('currentReputation', 0)) / 100.0,
                        safe_float(node_info.get('initialReputation', 0)) / 100.0,
                        # Attack detection features
                        attack_features['attack_interaction_ratio'],
                        behavioral_change_score,
                        attack_features['suspicious_pattern_count'],
                        attack_features['bribery_confidence'],
                        # Additional bribery-specific features
                        safe_float(attack_features.get('reputation_decline_rate', 0.0)),
                        safe_float(attack_features.get('decision_bias_score', 0.0)),
                        safe_float(attack_features.get('coordination_score', 0.0)),
                        safe_float(attack_features.get('economic_incentive_score', 0.0)),
                        safe_float(attack_features.get('timing_anomaly_score', 0.0)),
                    ]
                    
                    # Enhanced anomaly detection with ground truth
                    if node_address in ground_truth_labels.get('briberyDetection', {}):
                        is_bribery = ground_truth_labels['briberyDetection'][node_address]
                    else:
                        # Fallback heuristic
                        is_bribery = 1.0 if (behavioral_change_score > 0.5 or 
                                           attack_features['attack_interaction_ratio'] > 0.2) else 0.0
                    
                    processed_data['bribery_detection']['features'].append(features)
                    processed_data['bribery_detection']['labels'].append(is_bribery)
            
            # Remove duplicates for all models
            for model_name, model_data in processed_data.items():
                features_list = model_data['features']
                labels_list = model_data['labels']
                if len(features_list) == 0:
                    continue
                
                # Remove exact duplicates
                seen = set()
                unique_feats = []
                unique_labels = []
                for f, l in zip(features_list, labels_list):
                    f_tuple = tuple(f)
                    if f_tuple in seen:
                        dropped_interactions += 1
                        continue
                    seen.add(f_tuple)
                    unique_feats.append(f)
                    unique_labels.append(l)
                processed_data[model_name]['features'] = unique_feats
                processed_data[model_name]['labels'] = unique_labels
            
            # Enhanced synthetic data generation for attack patterns
            synthetic_data_generated = False
            MIN_SAMPLES_REQUIRED = 30  # Fixed minimum requirement for all models
            
            for model_name, model_data in processed_data.items():
                current_samples = len(model_data['features'])
                
                if current_samples < MIN_SAMPLES_REQUIRED:
                    needed_samples = MIN_SAMPLES_REQUIRED - current_samples
                    logger.info(f"{model_name}: Only {current_samples} real samples. Generating {needed_samples} enhanced synthetic samples.")
                    
                    synthetic_features, synthetic_labels = self._generate_enhanced_attack_synthetic_data(
                        model_name, model_data['features'], model_data['labels'], needed_samples, 
                        nodes_data, attack_metadata
                    )
                    
                    model_data['features'].extend(synthetic_features)
                    model_data['labels'].extend(synthetic_labels)
                    synthetic_data_generated = True
                    
                    logger.info(f"{model_name}: Now has {len(model_data['features'])} samples (real + enhanced synthetic)")

            if synthetic_data_generated:
                logger.info("Enhanced synthetic data generation with attack patterns completed.")
            
            # Final validation
            for model_name, model_data in processed_data.items():
                current_samples = len(model_data['features'])
                
                if current_samples < MIN_SAMPLES_REQUIRED:
                    logger.error(f"{model_name}: Still insufficient samples ({current_samples}), required minimum {MIN_SAMPLES_REQUIRED}.")
                    processed_data[model_name]['features'] = []
                    processed_data[model_name]['labels'] = []
                else:
                    logger.info(f"{model_name}: {current_samples} samples available (minimum {MIN_SAMPLES_REQUIRED} satisfied).")
            
            logger.info(f"Enhanced preprocessing completed: dropped {dropped_nodes} nodes and {dropped_interactions} duplicate interactions.")
            
            # Log attack data statistics
            if attack_metadata:
                total_positive_sybil = sum(processed_data['sybil_detection']['labels'])
                total_positive_bribery = sum(processed_data['bribery_detection']['labels'])
                logger.info(f"Attack detection summary: {total_positive_sybil} Sybil positives, {total_positive_bribery} bribery attacks detected")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in enhanced data preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
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
    
    def prepare_bribery_detection_data(self, 
                                      nodes_data: Dict[str, Any],
                                      transactions_data: Dict[str, List[Dict[str, Any]]],
                                      events_data: Dict[str, List[Dict[str, Any]]],
                                      time_window: int = 10,
                                      bribed_nodes: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for bribery detection model.
        
        Args:
            nodes_data: Dictionary of node data
            transactions_data: Dictionary of transaction data by node
            events_data: Dictionary of event data by node
            time_window: Number of time steps in the time series
            bribed_nodes: List of known bribed node addresses
            
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
                
                # Determine label (1 for bribed, 0 for normal)
                is_bribed = 0
                if bribed_nodes and node_address in bribed_nodes:
                    is_bribed = 1
                
                labels.append(is_bribed)
        
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
                         shuffle: bool = True):
        """
        Create TensorFlow dataset from features and labels.
        
        Args:
            features: Feature array
            labels: Label array
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
        Returns:
            TensorFlow dataset if TensorFlow is available, None otherwise
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create TF dataset.")
            return None
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))
        dataset = dataset.batch(batch_size)
        return dataset
    
    def create_federated_tf_datasets(self, 
                                    client_features: List[np.ndarray], 
                                    client_labels: List[np.ndarray], 
                                    batch_size: int = 32, 
                                    shuffle: bool = True, 
                                    random_seed: int = 42):
        """
        Create federated TensorFlow datasets for each client.
        Returns a list of tf.data.Dataset if TensorFlow is available, else None.
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create federated TF datasets.")
            return None
        federated_datasets = []
        for features, labels in zip(client_features, client_labels):
            client_dataset = self.create_tf_dataset(
                features, labels, batch_size=batch_size, shuffle=shuffle)
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
    
    def extract_bribery_detection_features(self, processed_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract features specifically for bribery detection model.
        
        Args:
            processed_data: Processed blockchain data
            
        Returns:
            Dictionary mapping node IDs to feature vectors
        """
        bribery_features = {}
        
        # Process node data
        for node_id, node_data in processed_data.get("nodes", {}).items():
            if "features" in node_data:
                feature_dict = node_data["features"]
                
                # Extract relevant features for bribery detection (15 features)
                relevant_features = []
                
                # Transaction patterns
                for key in ["tx_frequency_mean", "tx_frequency_std", "tx_value_mean", 
                           "tx_value_std", "tx_value_min", "tx_value_max", "tx_value_total"]:
                    relevant_features.append(feature_dict.get(key, 0))
                
                # Reputation and behavioral change indicators
                relevant_features.append(feature_dict.get("reputation", 0))
                
                # Bribery-specific features
                relevant_features.append(feature_dict.get("reputation_decline_rate", 0))
                relevant_features.append(feature_dict.get("decision_bias_score", 0))
                relevant_features.append(feature_dict.get("coordination_score", 0))
                relevant_features.append(feature_dict.get("economic_incentive_score", 0))
                relevant_features.append(feature_dict.get("timing_anomaly_score", 0))
                relevant_features.append(feature_dict.get("behavioral_change_score", 0))
                relevant_features.append(feature_dict.get("corruption_confidence", 0))
                
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

    def _generate_synthetic_blockchain_data(self, model_name: str, existing_features: List[List[float]], 
                                           existing_labels: List[float], needed_samples: int, 
                                           nodes_data: Dict[str, Any]) -> Tuple[List[List[float]], List[float]]:
        """
        Generate synthetic blockchain data that matches the structure and patterns of real data.
        Uses existing data statistics to create realistic synthetic samples.
        """
        import random
        import numpy as np
        
        synthetic_features = []
        synthetic_labels = []
        
        if not existing_features:
            # If no existing features, create baseline features based on model type
            if model_name == 'sybil_detection':
                for _ in range(needed_samples):
                    # Create realistic sybil detection features
                    features = [
                        random.uniform(0.3, 1.0),  # currentReputation
                        random.uniform(0.5, 1.0),  # initialReputation  
                        random.choice([0.0, 1.0]), # isVerified
                        random.uniform(0.1, 0.8),  # interactions normalized
                        random.uniform(0.0, 0.6),  # vote_actions
                        random.uniform(0.1, 0.7),  # transfer_actions
                        random.uniform(0.0, 0.4),  # production_actions
                        random.uniform(0.0, 0.4),  # marketplace_actions
                        random.uniform(0.0, 0.3),  # logistics_actions
                        random.uniform(0.0, 0.2),  # dispute_actions
                    ]
                    # Generate realistic label (mostly non-sybil)
                    label = 1.0 if random.random() < 0.15 else 0.0
                    synthetic_features.append(features)
                    synthetic_labels.append(label)
                    
            elif model_name == 'batch_monitoring':
                for _ in range(needed_samples):
                    features = [
                        random.uniform(0.2, 1.0),  # transactions normalized
                        random.uniform(0.4, 1.0),  # total_votes
                        random.uniform(0.6, 1.0),  # approval_rate
                        random.uniform(0.1, 0.8),  # processing_duration
                        random.choice([0.0, 1.0]), # is_committed
                        random.uniform(0.6, 1.0),  # validators normalized
                    ]
                    # Generate realistic anomaly label
                    label = 1.0 if random.random() < 0.2 else 0.0
                    synthetic_features.append(features)
                    synthetic_labels.append(label)
                    
            elif model_name == 'bribery_detection':
                for _ in range(needed_samples):
                    # Enhanced features for bribery detection (15 features)
                    features = [
                        random.uniform(0.1, 1.0),  # interactions normalized
                        random.uniform(0.1, 0.8),  # interaction_diversity
                        random.uniform(0.2, 2.0),  # avg_interval hours
                        random.uniform(0.5, 10.0), # min_interval minutes
                        random.uniform(0.4, 1.0),  # current_reputation
                        random.uniform(0.4, 1.0),  # initial_reputation
                        random.uniform(0.0, 0.3),  # attack_interaction_ratio
                        random.uniform(0.0, 0.4),  # behavioral_change_score
                        random.uniform(0.0, 0.2),  # suspicious_pattern_count
                        random.uniform(0.0, 0.3),  # bribery_confidence
                        random.uniform(0.0, 0.2),  # reputation_decline_rate
                        random.uniform(0.0, 0.3),  # decision_bias_score
                        random.uniform(0.0, 0.2),  # coordination_score
                        random.uniform(0.0, 0.3),  # economic_incentive_score
                        random.uniform(0.0, 0.2),  # timing_anomaly_score
                    ]
                    # Generate realistic bribery detection label
                    label = 1.0 if random.random() < 0.15 else 0.0
                    synthetic_features.append(features)
                    synthetic_labels.append(label)
            elif model_name == 'arbitrator_bias':
                for _ in range(needed_samples):
                    features = [
                        random.uniform(0.0, 1.0),  # arbitrator_vote_bias
                        random.uniform(0.0, 1.0),  # dispute_decision_bias
                        random.uniform(0.0, 1.0),  # batch_approval_bias
                        random.uniform(0.0, 1.0),  # batch_denial_bias
                        random.uniform(0.0, 1.0),  # node_reputation
                        random.uniform(0.0, 1.0),  # governance_participation
                        random.uniform(0.0, 1.0),  # arbitrator_activity
                        random.uniform(0.0, 1.0),  # dispute_frequency
                        random.uniform(0.0, 1.0),  # transfer_frequency
                        random.uniform(0.0, 1.0),  # vote_inconsistency
                    ]
                    label = 1.0 if random.random() < 0.1 else 0.0
                    synthetic_features.append(features)
                    synthetic_labels.append(label)
            elif model_name == 'dispute_risk':
                for _ in range(needed_samples):
                    features = [
                        random.uniform(0.0, 1.0),  # dispute_frequency
                        random.uniform(0.0, 1.0),  # dispute_resolution_time
                        random.uniform(0.0, 1.0),  # dispute_decision_bias
                        random.uniform(0.0, 1.0),  # batch_approval_bias
                        random.uniform(0.0, 1.0),  # batch_denial_bias
                        random.uniform(0.0, 1.0),  # node_reputation
                        random.uniform(0.0, 1.0),  # governance_participation
                        random.uniform(0.0, 1.0),  # arbitrator_activity
                        random.uniform(0.0, 1.0),  # transfer_frequency
                        random.uniform(0.0, 1.0),  # vote_inconsistency
                    ]
                    label = 1.0 if random.random() < 0.15 else 0.0
                    synthetic_features.append(features)
                    synthetic_labels.append(label)
        else:
            # Use existing data statistics to generate similar synthetic data
            existing_array = np.array(existing_features)
            existing_labels_array = np.array(existing_labels)
            
            # Calculate statistics from existing data
            means = np.mean(existing_array, axis=0)
            stds = np.std(existing_array, axis=0)
            
            # Calculate label distribution
            positive_ratio = np.mean(existing_labels_array) if len(existing_labels_array) > 0 else 0.2
            
            for _ in range(needed_samples):
                # Generate features based on existing statistics with some noise
                features = []
                for i in range(len(means)):
                    # Add controlled noise to maintain realistic ranges
                    noise_factor = random.uniform(0.8, 1.2)
                    std_factor = max(0.1, stds[i]) * noise_factor
                    feature_value = random.gauss(means[i], std_factor)
                    
                    # Ensure realistic bounds based on feature type
                    if i < 3:  # Reputation and verification features
                        feature_value = max(0.0, min(1.0, feature_value))
                    elif i < 6:  # Normalized count features
                        feature_value = max(0.0, min(1.0, feature_value))
                    else:  # Other features
                        feature_value = max(0.0, feature_value)
                    
                    features.append(feature_value)
                
                # Generate label based on existing distribution
                label = 1.0 if random.random() < positive_ratio else 0.0
                
                synthetic_features.append(features)
                synthetic_labels.append(label)
        
        logger.info(f"Generated {len(synthetic_features)} synthetic samples for {model_name}")
        return synthetic_features, synthetic_labels

    def _generate_enhanced_attack_synthetic_data(self, model_name: str, existing_features: List[List[float]], 
                                               existing_labels: List[float], needed_samples: int, 
                                               nodes_data: Dict[str, Any], attack_metadata: Dict[str, Any]) -> Tuple[List[List[float]], List[float]]:
        """
        Generate enhanced synthetic data that includes attack patterns and realistic blockchain behaviors.
        This ensures the FL models have sufficient data to detect both normal and attack scenarios.
        """
        import random
        import numpy as np
        
        synthetic_features = []
        synthetic_labels = []
        
        logger.info(f"Generating enhanced synthetic data for {model_name} with attack pattern awareness")
        
        # Calculate label distribution from existing data (if any)
        if existing_labels:
            positive_ratio = np.mean(existing_labels)
        else:
            # Default ratios based on realistic attack scenarios
            if model_name == 'sybil_detection':
                positive_ratio = 0.15  # 15% attack nodes
            elif model_name == 'bribery_detection':
                positive_ratio = 0.12  # 12% bribery attacks
            else:
                positive_ratio = 0.10  # 10% anomalies
        
        # Enhanced feature generation based on model type
        for i in range(needed_samples):
            if model_name == 'sybil_detection':
                # Generate Sybil detection features with attack awareness
                is_attack = random.random() < positive_ratio
                
                if is_attack:
                    # Generate attack pattern features
                    features = [
                        random.uniform(0.3, 0.7),  # currentReputation (lower for attacks)
                        random.uniform(0.5, 0.9),  # initialReputation (artificially high)
                        1.0,  # isVerified (attackers often verify first)
                        random.uniform(0.1, 0.3),  # interactions (fewer genuine interactions)
                        random.uniform(0.6, 1.0),  # vote_actions (excessive voting)
                        random.uniform(0.3, 0.8),  # transfer_actions 
                        random.uniform(0.0, 0.2),  # production_actions (low legitimate activity)
                        random.uniform(0.0, 0.3),  # marketplace_actions
                        random.uniform(0.0, 0.2),  # logistics_actions
                        random.uniform(0.1, 0.5),  # dispute_actions
                        random.uniform(0.2, 0.8),  # attack_sybil_actions (attack-specific)
                        random.uniform(0.4, 0.9),  # suspicious_pattern_count
                        random.uniform(0.3, 0.7),  # attack_interaction_ratio
                        random.uniform(0.7, 0.95), # sybil_confidence
                    ]
                    label = 1.0
                else:
                    # Generate legitimate node features
                    features = [
                        random.uniform(0.6, 1.0),  # currentReputation (higher for legitimate)
                        random.uniform(0.6, 1.0),  # initialReputation
                        random.choice([0.0, 1.0]), # isVerified
                        random.uniform(0.2, 0.8),  # interactions (varied)
                        random.uniform(0.0, 0.4),  # vote_actions (normal voting)
                        random.uniform(0.1, 0.6),  # transfer_actions
                        random.uniform(0.1, 0.7),  # production_actions (legitimate activity)
                        random.uniform(0.1, 0.6),  # marketplace_actions
                        random.uniform(0.0, 0.5),  # logistics_actions
                        random.uniform(0.0, 0.2),  # dispute_actions (few disputes)
                        0.0,  # attack_sybil_actions (no attack actions)
                        random.uniform(0.0, 0.3),  # suspicious_pattern_count (low)
                        random.uniform(0.0, 0.1),  # attack_interaction_ratio (very low)
                        0.0,  # sybil_confidence (not sybil)
                    ]
                    label = 0.0
                    
            elif model_name == 'batch_monitoring':
                # Generate batch monitoring features
                is_anomalous = random.random() < positive_ratio
                
                if is_anomalous:
                    # Anomalous batch features
                    features = [
                        random.uniform(0.1, 0.4),  # transactions (few transactions)
                        random.uniform(0.2, 0.6),  # total_votes
                        random.uniform(0.0, 0.5),  # approval_rate (low approval)
                        random.uniform(0.6, 1.0),  # processing_duration (long duration)
                        0.0,  # is_committed (not committed)
                        random.uniform(0.2, 0.6),  # validators
                        random.uniform(0.2, 0.8),  # sybil_votes_ratio
                        random.uniform(0.1, 0.6),  # bribed_votes_ratio
                    ]
                    label = 1.0
                else:
                    # Normal batch features
                    features = [
                        random.uniform(0.4, 1.0),  # transactions
                        random.uniform(0.6, 1.0),  # total_votes
                        random.uniform(0.7, 1.0),  # approval_rate (high approval)
                        random.uniform(0.1, 0.5),  # processing_duration (normal)
                        1.0,  # is_committed
                        random.uniform(0.6, 1.0),  # validators
                        0.0,  # sybil_votes_ratio (no sybil votes)
                        0.0,  # bribed_votes_ratio (no bribed votes)
                    ]
                    label = 0.0
                    
            elif model_name == 'bribery_detection':
                # Generate bribery detection features (15 features)
                is_bribed = random.random() < positive_ratio
                
                if is_bribed:
                    # Bribery attack features
                    features = [
                        random.uniform(0.3, 0.8),  # interactions
                        random.uniform(0.1, 0.5),  # interaction_diversity (low diversity)
                        random.uniform(0.0, 0.2),  # avg_interval (too fast)
                        random.uniform(0.0, 0.1),  # min_interval (very fast)
                        random.uniform(0.2, 0.6),  # current_reputation (degraded)
                        random.uniform(0.5, 0.9),  # initial_reputation (was higher)
                        random.uniform(0.3, 0.8),  # attack_interaction_ratio
                        random.uniform(0.6, 1.0),  # behavioral_change_score
                        random.uniform(0.4, 0.9),  # suspicious_pattern_count
                        random.uniform(0.5, 0.9),  # bribery_confidence
                        random.uniform(0.3, 0.7),  # reputation_decline_rate
                        random.uniform(0.4, 0.8),  # decision_bias_score
                        random.uniform(0.3, 0.7),  # coordination_score
                        random.uniform(0.4, 0.8),  # economic_incentive_score
                        random.uniform(0.3, 0.6),  # timing_anomaly_score
                    ]
                    label = 1.0
                else:
                    # Normal behavior features (15 features)
                    features = [
                        random.uniform(0.3, 0.9),  # interactions
                        random.uniform(0.4, 0.9),  # interaction_diversity (good diversity)
                        random.uniform(0.2, 2.0),  # avg_interval (normal timing)
                        random.uniform(0.5, 10.0), # min_interval (reasonable gaps)
                        random.uniform(0.6, 1.0),  # current_reputation (good)
                        random.uniform(0.6, 1.0),  # initial_reputation (consistent)
                        random.uniform(0.0, 0.2),  # attack_interaction_ratio (low)
                        random.uniform(0.0, 0.3),  # behavioral_change_score (stable)
                        random.uniform(0.0, 0.3),  # suspicious_pattern_count (low)
                        0.0,  # bribery_confidence (not bribed)
                        random.uniform(0.0, 0.1),  # reputation_decline_rate (stable)
                        random.uniform(0.0, 0.2),  # decision_bias_score (fair)
                        random.uniform(0.0, 0.2),  # coordination_score (independent)
                        random.uniform(0.0, 0.2),  # economic_incentive_score (honest)
                        random.uniform(0.0, 0.2),  # timing_anomaly_score (normal)
                    ]
                    label = 0.0
            else:
                # Fallback for unknown model types
                features = [random.uniform(0.0, 1.0) for _ in range(8)]
                label = 1.0 if random.random() < positive_ratio else 0.0
            
            synthetic_features.append(features)
            synthetic_labels.append(label)
        
        logger.info(f"Generated {len(synthetic_features)} enhanced synthetic samples for {model_name} "
                   f"with {sum(synthetic_labels)} positive (attack) cases")
        return synthetic_features, synthetic_labels

    def create_synthetic_blockchain_context(self, base_samples: int = 35) -> Dict[str, Any]:
        """
        Create synthetic blockchain context data that matches demo_context.json format.
        This can be used to supplement real data for FL training.
        """
        import random
        import time
        
        # Blockchain interaction types with realistic frequencies
        interaction_types = [
            'MintProduct', 'ListProduct', 'TransferNFT', 'ReceiveNFT',
            'VoteBatch', 'VoteForArbitrator', 'StartTransport', 'CompleteTransport',
            'FinalizePurchase', 'TransferNFTViaBatch', 'ReceiveNFTViaBatch',
            'CommitBatchAttempt', 'ProposeBatch', 'MakeDisputeDecision'
        ]
        
        # Node roles and types
        roles = [(0, "Manufacturer", "Primary"), (1, "Transporter", "Secondary"), 
                (2, "Retailer", "Primary"), (3, "Consumer", "Secondary")]
        
        synthetic_context = {
            "contractAddress": f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            "nodes": {},
            "products": {},
            "batches": {},
            "disputes": {}
        }
        
        current_time = int(time.time())
        
        # Generate synthetic nodes
        for i in range(base_samples):
            node_address = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"
            role_idx = random.randint(0, 3)
            role, role_name, node_type_name = roles[role_idx]
            
            # Generate realistic interactions
            num_interactions = random.randint(3, 25)
            interactions = []
            
            for j in range(num_interactions):
                interaction_type = random.choice(interaction_types)
                timestamp = current_time - random.randint(0, 30*24*3600)  # Last 30 days
                
                interaction = {
                    "type": interaction_type,
                    "timestamp": timestamp
                }
                
                # Add type-specific details
                if interaction_type in ['MintProduct', 'ListProduct', 'TransferNFT', 'ReceiveNFT']:
                    interaction["tokenId"] = str(random.randint(1, 1000))
                if interaction_type in ['VoteBatch', 'ProposeBatch']:
                    interaction["batchId"] = str(random.randint(1, 100))
                    if interaction_type == 'VoteBatch':
                        interaction["vote"] = random.choice([True, False])
                if interaction_type == 'ListProduct':
                    interaction["price"] = f"{random.uniform(0.1, 2.0):.1f} ETH"
                
                interaction["details"] = f"Synthetic {interaction_type} operation"
                interactions.append(interaction)
            
            # Sort interactions by timestamp
            interactions.sort(key=lambda x: x['timestamp'])
            
            synthetic_context["nodes"][node_address] = {
                "address": node_address,
                "name": f"Synthetic_{role_name}_{i+1}",
                "role": role,
                "roleName": role_name,
                "nodeType": 0 if node_type_name == "Primary" else 1,
                "nodeTypeName": node_type_name,
                "initialReputation": random.randint(80, 100),
                "currentReputation": random.randint(60, 100),
                "isVerified": random.choice([True, False]),
                "interactions": interactions
            }
        
        logger.info(f"Created synthetic blockchain context with {len(synthetic_context['nodes'])} nodes")
        return synthetic_context
