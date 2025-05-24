#!/usr/bin/env python3
"""
Federated training for Sybil detection in ChainFLIP system.

This module provides the federated training process for the Sybil detection model.
"""

import tensorflow_federated as tff
import tensorflow as tf
from model_definition import tff_model_fn  # Import from our model_definition module

def build_fed_avg_process():
    """
    Builds the Federated Averaging iterative process.
    
    Returns:
        Federated Averaging process
    """
    # Create the FedAvg process with TFF's built-in optimizer functions
    fed_avg_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=tff_model_fn,
        client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.1),
        server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    )
    return fed_avg_process

if __name__ == '__main__':
    print("Building Federated Averaging process...")
    iterative_process = build_fed_avg_process()
    print("Federated Averaging process built successfully.")
    print("Initialize signature:", iterative_process.initialize.type_signature)
    print("Next signature:", iterative_process.next.type_signature)
