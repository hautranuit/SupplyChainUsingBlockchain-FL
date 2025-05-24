#!/usr/bin/env python3
"""
Real data preparation for Node Behavior Timeseries model in the ChainFLIP system.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple

def generate_timeseries_features(is_anomalous: bool, num_features: int = 10, random_seed: int = None) -> np.ndarray:
    """
    Generate synthetic timeseries features for node behavior analysis.
    
    Args:
        is_anomalous: Whether this behavior is anomalous
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
    
    # Make anomalous behaviors have distinctive patterns
    if is_anomalous:
        # Features indicating anomalies
        features[0] = np.random.normal(1.8, 0.3)  # High transaction frequency
        features[1] = np.random.normal(1.5, 0.4)  # High deviation from past patterns
        features[2] = np.random.normal(0.2, 0.1)  # Low consistency score
        features[3] = np.random.normal(1.7, 0.3)  # High similarity to known attack patterns
    else:
        # Normal behavior features
        features[0] = np.random.normal(0.5, 0.2)  # Normal transaction frequency
        features[1] = np.random.normal(0.6, 0.2)  # Normal deviation from past patterns
        features[2] = np.random.normal(0.8, 0.1)  # Normal consistency score
        features[3] = np.random.normal(0.4, 0.2)  # Low similarity to known attack patterns
    
    return features

def make_federated_data_p3_timeseries_real(
    clients_info: List[Dict[str, Any]],
    sybil_attack_log: Optional[Dict[str, Any]] = None,
    samples_per_client: int = 50,
    num_features: int = 10
) -> List[tf.data.Dataset]:
    """
    Create federated data for Node Behavior Timeseries model.
    
    Args:
        clients_info: List of client information with address, role, etc.
        sybil_attack_log: Sybil attack log data from sybil_attack_log.json
        samples_per_client: Number of samples per client
        num_features: Number of features per sample
        
    Returns:
        List of TensorFlow datasets
    """
    # Get sybil and bribed node addresses from attack log
    sybil_addresses = []
    bribed_addresses = []
    
    if sybil_attack_log:
        # Extract sybil node addresses
        if "sybilNodes" in sybil_attack_log:
            for sybil_node in sybil_attack_log["sybilNodes"]:
                if "address" in sybil_node:
                    sybil_addresses.append(sybil_node["address"])
        
        # Extract bribed node addresses
        if "scenarioD" in sybil_attack_log and "bribedNodes" in sybil_attack_log["scenarioD"]:
            for bribed_node in sybil_attack_log["scenarioD"]["bribedNodes"]:
                if "address" in bribed_node:
                    bribed_addresses.append(bribed_node["address"])
    
    # Create federated datasets
    federated_data = []
    
    # Generate data for each client
    for client_id, client_info in enumerate(clients_info):
        client_address = client_info.get("address", "")
        client_role = client_info.get("role", "Unknown")
        
        # Generate features and labels
        features_list = []
        labels_list = []
        
        # Determine if this client is suspicious
        is_sybil = client_address in sybil_addresses
        is_bribed = client_address in bribed_addresses
        is_suspicious = is_sybil or is_bribed
        
        # Determine anomaly rate based on client suspiciousness
        anomaly_rate = 0.8 if is_suspicious else 0.05
        
        # Generate samples for this client
        for i in range(samples_per_client):
            # Randomly determine if this sample is for anomalous behavior
            sample_is_anomalous = random.random() < anomaly_rate
            
            # Generate features
            sample_features = generate_timeseries_features(sample_is_anomalous, num_features, client_id * 1000 + i)
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
