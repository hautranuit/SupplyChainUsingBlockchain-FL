#!/usr/bin/env python3
"""
Real data preparation for Dispute Risk model in the ChainFLIP system.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple

def generate_dispute_features(is_dispute_risk: bool, num_features: int = 10, random_seed: int = None) -> np.ndarray:
    """
    Generate synthetic features for dispute risk analysis.
    
    Args:
        is_dispute_risk: Whether this transaction has high dispute risk
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
    
    # Make high-risk transactions have distinctive patterns
    if is_dispute_risk:
        # Features indicating dispute risk
        features[0] = np.random.normal(1.8, 0.3)  # High price deviation
        features[1] = np.random.normal(1.5, 0.4)  # High seller risk score
        features[2] = np.random.normal(0.2, 0.1)  # Low transaction transparency
        features[3] = np.random.normal(1.7, 0.3)  # High product category risk
    else:
        # Low-risk transaction features
        features[0] = np.random.normal(0.5, 0.2)  # Normal price deviation
        features[1] = np.random.normal(0.6, 0.2)  # Normal seller risk score
        features[2] = np.random.normal(0.8, 0.1)  # Normal transaction transparency
        features[3] = np.random.normal(0.4, 0.2)  # Normal product category risk
    
    return features

def make_federated_data_p3_dispute_real(
    all_node_addresses: List[str],
    sybil_attack_log: Optional[Dict[str, Any]] = None,
    samples_per_client: int = 50,
    num_features: int = 10
) -> List[tf.data.Dataset]:
    """
    Create federated data for Dispute Risk model.
    
    Args:
        all_node_addresses: List of all node addresses
        sybil_attack_log: Sybil attack log data from sybil_attack_log.json
        samples_per_client: Number of samples per client
        num_features: Number of features per sample
        
    Returns:
        List of TensorFlow datasets
    """
    # Check if attack is active
    attack_active = False
    if sybil_attack_log:
        # Check for specific dispute-related attacks
        if "scenarioD" in sybil_attack_log and "actions" in sybil_attack_log["scenarioD"]:
            for action in sybil_attack_log["scenarioD"]["actions"]:
                if action.get("type") in ["BypassVerification", "FalseQualityReport", "ApproveCounterfeit"]:
                    attack_active = True
                    break
    
    # Create federated datasets
    federated_data = []
    
    # Generate data for each node as a client
    for client_id, client_address in enumerate(all_node_addresses):
        # Generate features and labels
        features_list = []
        labels_list = []
        
        # Determine dispute risk rate based on attack presence
        dispute_rate = 0.8 if attack_active else 0.05
        
        # Generate samples for this client
        for i in range(samples_per_client):
            # Randomly determine if this sample has high dispute risk
            sample_is_high_risk = random.random() < dispute_rate
            
            # Generate features
            sample_features = generate_dispute_features(sample_is_high_risk, num_features, client_id * 1000 + i)
            features_list.append(sample_features)
            labels_list.append(1 if sample_is_high_risk else 0)
        
        # Convert to NumPy arrays
        features_array = np.array(features_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.float32).reshape(-1, 1)
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
        dataset = dataset.batch(10).repeat(10)  # Batch size and repeat count
        
        federated_data.append(dataset)
    
    return federated_data
