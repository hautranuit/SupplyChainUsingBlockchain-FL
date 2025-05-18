import tensorflow as tf
import numpy as np
import os
from model_definition import create_keras_model
from real_data_preparation_sybil import NUM_SYBIL_FEATURES

class SybilDetectionModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = create_keras_model()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc")
                ]
            )

    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead")
            self.model = create_keras_model()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc")
                ]
            )

    def save_model(self, model_path):
        """Save the trained model to disk"""
        try:
            self.model.save(model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def predict(self, features):
        """Make prediction using the model"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Ensure features are in the correct shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)
        return prediction[0][0]  # Return probability of being Sybil

    def predict_batch(self, features_batch):
        """Make predictions for a batch of features"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Make predictions
        predictions = self.model.predict(features_batch, verbose=0)
        return predictions.flatten()  # Return array of probabilities 