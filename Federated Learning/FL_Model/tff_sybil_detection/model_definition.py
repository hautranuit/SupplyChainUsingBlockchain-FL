#!/usr/bin/env python3
"""
Model definition for Sybil detection in ChainFLIP system.

This module provides the TensorFlow model for detecting Sybil nodes in the network.
"""

import tensorflow as tf
import tensorflow_federated as tff

def create_keras_model():
    """
    Create a Keras model for enhanced Sybil detection with 25 features.
    
    Returns:
        A compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(25,)),  # Increased for 25 features
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def tff_model_fn():
    """
    Create a TFF model for Sybil detection.
    
    Returns:
        A TFF model
    """
    # Create a Keras model
    keras_model = create_keras_model()
      # Convert to TFF model
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=tf.TensorSpec(shape=(None, 25), dtype=tf.float32),  # Updated to 25 features
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )
