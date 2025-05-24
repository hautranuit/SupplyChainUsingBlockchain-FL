#!/usr/bin/env python3
"""
Real data preparation for Batch Monitoring model in the ChainFLIP system.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple

def generate_batch_features(is_anomalous: bool, num_features: int = 10, random_seed: int = None) -> np.ndarray:
    """
    Generate synthetic features for batch monitoring.
    
    Args:
        is_anomalous: Whether this batch has anomalies
        num_features: Number of features to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        NumPy array of features
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Generate random features
    features = np.random.normal(size=num_features)
    
    # Make anomalous batches have distinctive patterns
    if is_anomalous:
        # Features indicating anomalies
        features[0] = np.random.normal(1.8, 0.3)  # High temperature variation
        features[1] = np.random.normal(1.5, 0.4)  # High humidity variation
        features[2] = np.random.normal(0.2, 0.1)  # Low quality score
        features[3] = np.random.normal(1.7, 0.3)  # High time deviation
    else:
        # Normal batch features
        features[0] = np.random.normal(0.5, 0.2)  # Normal temperature variation
        features[1] = np.random.normal(0.6, 0.2)  # Normal humidity variation
        features[2] = np.random.normal(0.8, 0.1)  # Normal quality score
        features[3] = np.random.normal(0.4, 0.2)  # Normal time deviation
    
    return features

def make_federated_data_batch_monitoring_real(
    all_node_addresses: List[str],
    num_fl_clients: int = 3,
    sybil_attack_log: Optional[Dict[str, Any]] = None,
    samples_per_client: int = 50,
    num_features: int = 10
) -> List[tf.data.Dataset]:
    """
    Create federated data for Batch Monitoring model.
    
    Args:
        all_node_addresses: List of all node addresses
        num_fl_clients: Number of FL clients to generate data for
        sybil_attack_log: Sybil attack log data from sybil_attack_log.json
        samples_per_client: Number of samples per client
        num_features: Number of features per sample
        
    Returns:
        List of TensorFlow datasets
    """
    # Check if attack is active
    attack_active = False
    if sybil_attack_log:
        # Check for specific batch-related attacks
        if "scenarioC" in sybil_attack_log and "actions" in sybil_attack_log["scenarioC"]:
            for action in sybil_attack_log["scenarioC"]["actions"]:
                if action.get("type") in ["BatchProcessing", "QualityControl"]:
                    attack_active = True
                    break
    
    # Create federated datasets
    federated_data = []
    
    # Determine FL client assignments
    client_addresses = []
    if len(all_node_addresses) <= num_fl_clients:
        # Each node is its own client
        client_addresses = [addr for addr in all_node_addresses]
    else:
        # Randomly select nodes for each client
        client_addresses = random.sample(all_node_addresses, num_fl_clients)
    
    # Generate data for each client
    for client_id, client_address in enumerate(client_addresses):
        # Generate features and labels
        features_list = []
        labels_list = []
        
        # Determine anomaly rate based on attack presence
        anomaly_rate = 0.7 if attack_active else 0.05
        
        # Generate samples for this client
        for i in range(samples_per_client):
            # Randomly determine if this sample is for an anomalous batch
            sample_is_anomalous = random.random() < anomaly_rate
            
            # Generate features
            sample_features = generate_batch_features(sample_is_anomalous, num_features, client_id * 1000 + i)
            features_list.append(sample_features)
            labels_list.append(1 if sample_is_anomalous else 0)
        
        # Convert to NumPy arrays
        features_array = np.array(features_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.float32).reshape(-1, 1)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
        dataset = dataset.batch(10).repeat(10)  # Batch size and repeat count
        
        federated_data.append(dataset)
    
    return federated_data
