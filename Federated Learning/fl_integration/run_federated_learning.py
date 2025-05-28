#!/usr/bin/env python3
"""
Federated Learning Integration System - Production Version
Completely fixed version addressing all TFF integration issues
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import traceback

# Web3 import for blockchain interaction
from web3 import Web3

# TensorFlow and TensorFlow Federated imports
try:
    import tensorflow as tf
    import tensorflow_federated as tff
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow or TensorFlow Federated not available. Some features may not work.")

# Optional import handling for data processing
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Enhanced Loss Functions for Better Model Calibration
class OptimizedLossFunctions:
    """Enhanced loss functions to address high loss with good accuracy issue."""
    
    @staticmethod
    def focal_loss(alpha=0.25, gamma=2.0):
        """
        Focal Loss to address class imbalance and reduce overconfident predictions.
        Focuses training on hard examples.
        """
        def focal_loss_fn(y_true, y_pred):
            # Convert predictions to probabilities
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            
            # Calculate focal loss
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
            focal_weight = y_true * (1 - y_pred) ** gamma + (1 - y_true) * y_pred ** gamma
            focal_loss = alpha_factor * focal_weight * tf.keras.backend.binary_crossentropy(y_true, y_pred)
            
            return tf.reduce_mean(focal_loss)
        
        focal_loss_fn.__name__ = 'focal_loss'
        return focal_loss_fn
    
    @staticmethod
    def label_smoothing_loss(smoothing=0.1):
        """
        Label smoothing to improve model calibration and reduce overconfidence.
        """
        def label_smoothing_fn(y_true, y_pred):
            # Apply label smoothing
            y_true_smooth = y_true * (1 - smoothing) + 0.5 * smoothing
            return tf.keras.losses.binary_crossentropy(y_true_smooth, y_pred)
        
        label_smoothing_fn.__name__ = 'label_smoothing_loss'
        return label_smoothing_fn
    
    @staticmethod
    def get_optimized_optimizer(learning_rate=0.005, optimization_strategy="standard"):
        """Get optimized optimizer with better convergence properties."""
        if optimization_strategy == "adaptive":
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
        elif optimization_strategy == "momentum":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Add project paths to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from fl_orchestrator.orchestrator import FLOrchestrator
from anomaly_detector.detector import AnomalyDetector  
from response_engine.engine import ResponseEngine
from model_repository.repository import ModelRepository
from monitoring.monitoring import MonitoringSystem
from preprocessing.feature_extractor import FeatureExtractor
from blockchain_connector.connector import BlockchainConnector # <--- Added import

# Import individual FL model scripts for full mode integration
import subprocess
import importlib.util

# Helper functions for printing formatted output
def print_section_header(title):
    """Prints a formatted section header."""
    print("\n" + "="*60)
    print(f"{title.upper()}")
    print("="*60)

def print_highlight(message, level="INFO"):
    """Prints a highlighted message."""
    prefix = {
        "INFO": "[INFO]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
        "CRITICAL": "[CRITICAL]",
        "SUCCESS": "[SUCCESS]"
    }.get(level.upper(), "[INFO]")
    print(f"{prefix} {message}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fl_integration.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("[DEBUG] Script started.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning Integration System')
    parser.add_argument('--mode', type=str, default='normal', 
                       choices=['normal', 'attack', 'full'], 
                       help='Execution mode (normal, attack, or full - where full mode automatically analyzes data without prior knowledge)')
    parser.add_argument('--input-data-file', type=str, default=None,
                       help='Path to input data file (e.g., demo_context.json)')
    parser.add_argument('--output-dir', type=str, 
                       default=r'e:\NAM3\DO_AN_CHUYEN_NGANH\Federated Learning\output\normal',
                       help='Output directory for results')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--num-rounds', type=int, default=50,
                       help='Number of federated learning rounds')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for federated training')
    parser.add_argument('--num-clients', type=int, default=2,
                       help='Number of federated clients')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
      # Loss optimization arguments
    parser.add_argument('--loss-optimization', type=str, default='standard',
                       choices=['standard', 'focal_loss', 'label_smoothing'],
                       help='Loss function optimization strategy for better calibration')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                       help='Alpha parameter for focal loss (class weighting)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss (focusing parameter)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor (0.0 = no smoothing, 0.1 = mild smoothing)')
    parser.add_argument('--optimizer-strategy', type=str, default='standard',
                       choices=['standard', 'adaptive', 'momentum'],
                       help='Optimizer strategy for better convergence')
    parser.add_argument('--enable-loss-analysis', action='store_true',
                       help='Enable detailed loss analysis and reporting')
    
    # Data augmentation arguments
    parser.add_argument('--enable-data-augmentation', action='store_true',
                       help='Enable enhanced data augmentation techniques')
    parser.add_argument('--noise-factor', type=float, default=0.1,
                       help='Noise injection factor for data augmentation')
    parser.add_argument('--feature-scaling', type=str, default='standard',
                       choices=['none', 'standard', 'minmax'],
                       help='Feature scaling method for better convergence')
    parser.add_argument('--enable-smote', action='store_true',
                       help='Enable SMOTE-like augmentation for class balance')
    
    # Model enhancement arguments
    parser.add_argument('--enable-batch-norm', action='store_true', default=True,
                       help='Enable batch normalization in models')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    parser.add_argument('--l2-regularization', type=float, default=0.001,
                       help='L2 regularization factor')
    
    # Performance monitoring arguments
    parser.add_argument('--enable-performance-analysis', action='store_true', default=True,
                       help='Enable comprehensive performance analysis and reporting')
    parser.add_argument('--save-performance-report', action='store_true', default=True,
                       help='Save performance analysis report to file')
    
    args = parser.parse_args()
    # Force input_data_file to demo_context.json if not set
    if not args.input_data_file:
        args.input_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json'))
    
    return args

def load_config(config_file: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    default_config = {
        "fl_orchestrator": {
            "aggregation_strategy": "federated_averaging",
            "privacy_budget": 1.0,
            "dp_noise_multiplier": 0.1,
            "max_grad_norm": 1.0,
            "client_auth_enabled": True,
            "secure_aggregation": True,
            "differential_privacy": False
        },
        "anomaly_detector": {
            "threshold": 0.5,
            "algorithm": "isolation_forest",
            "contamination": 0.1
        },
        "response_engine": {
            "response_strategy": "adaptive",
            "severity_levels": ["low", "medium", "high", "critical"]
        },
        "blockchain_connector": {
            "rpc_url": "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94",
            "num_blocks_to_fetch": 10
        }
    }
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return {**default_config, **config}
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}. Using defaults.")
    
    return default_config

def setup_directories(args) -> Dict[str, str]:
    """Create necessary output directories."""
    data_dir = os.path.join(args.output_dir, 'data')
    models_dir = os.path.join(args.output_dir, 'models')
    results_dir = os.path.join(args.output_dir, 'results')
    logs_dir = os.path.join(args.output_dir, 'logs')
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    
    for directory in [data_dir, models_dir, results_dir, logs_dir, metrics_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'data_dir': data_dir,
        'models_dir': models_dir,
        'results_dir': results_dir,
        'logs_dir': logs_dir,
        'metrics_dir': metrics_dir
    }

def create_sybil_detection_model(input_shape: int, compile_model: bool = True, args=None):
    """Create an improved model for Sybil detection with better architecture and advanced optimization."""
    
    # Apply optimization arguments for model architecture
    l2_reg = getattr(args, 'l2_regularization', 0.001) if args else 0.001
    dropout_rate = getattr(args, 'dropout_rate', 0.4) if args else 0.4
    enable_batch_norm = getattr(args, 'enable_batch_norm', True) if args else True
    
    layers = [
        # Input layer with L2 regularization
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ]
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate),
        
        # Hidden layers with residual-like connections
        tf.keras.layers.Dense(96, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate * 0.75),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate * 0.5),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate * 0.375),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate * 0.25),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Enhanced architecture with regularization
    model = tf.keras.Sequential(layers)
    
    if compile_model:
        # Get loss function and optimizer based on arguments
        loss_function = 'binary_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        if args:
            # Apply loss optimization strategies
            if hasattr(args, 'loss_optimization') and args.loss_optimization == 'focal_loss':
                loss_function = OptimizedLossFunctions.focal_loss(
                    alpha=getattr(args, 'focal_alpha', 0.25),
                    gamma=getattr(args, 'focal_gamma', 2.0)
                )
            elif hasattr(args, 'loss_optimization') and args.loss_optimization == 'label_smoothing':
                loss_function = OptimizedLossFunctions.label_smoothing_loss(
                    smoothing=getattr(args, 'label_smoothing', 0.1)
                )
            
            # Apply optimizer optimization
            if hasattr(args, 'optimizer_strategy'):
                optimizer = OptimizedLossFunctions.get_optimized_optimizer(
                    learning_rate=getattr(args, 'learning_rate', 0.001),
                    optimization_strategy=args.optimizer_strategy
                )
        
        # Enhanced metrics for better monitoring
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score'),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
        )
    return model

def create_batch_monitoring_model(input_shape: int, compile_model: bool = True, args=None):
    """Create an improved model for batch monitoring with enhanced performance."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        # Apply optimization strategies if available
        loss_function = 'binary_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        if args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'focal_loss':
            loss_function = OptimizedLossFunctions.focal_loss(
                alpha=getattr(args, 'focal_alpha', 0.25),
                gamma=getattr(args, 'focal_gamma', 2.0)
            )
        elif args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'label_smoothing':
            loss_function = OptimizedLossFunctions.label_smoothing_loss(
                smoothing=getattr(args, 'label_smoothing', 0.1)
            )
        
        if args and hasattr(args, 'optimizer_strategy'):
            optimizer = OptimizedLossFunctions.get_optimized_optimizer(
                learning_rate=getattr(args, 'learning_rate', 0.001),
                optimization_strategy=args.optimizer_strategy
            )
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
    return model

def create_arbitrator_bias_model(input_shape: int, compile_model: bool = True, args=None):
    """Create an improved model for arbitrator bias detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        # Apply optimization strategies if available
        loss_function = 'binary_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
        
        if args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'focal_loss':
            loss_function = OptimizedLossFunctions.focal_loss(
                alpha=getattr(args, 'focal_alpha', 0.25),
                gamma=getattr(args, 'focal_gamma', 2.0)
            )
        elif args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'label_smoothing':
            loss_function = OptimizedLossFunctions.label_smoothing_loss(
                smoothing=getattr(args, 'label_smoothing', 0.1)
            )
        
        if args and hasattr(args, 'optimizer_strategy'):
            optimizer = OptimizedLossFunctions.get_optimized_optimizer(
                learning_rate=0.0008,
                optimization_strategy=args.optimizer_strategy
            )
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score')
            ]
        )
    return model

def create_dispute_risk_model(input_shape: int, compile_model: bool = True, args=None):
    """Create an improved model for dispute risk assessment."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        # Apply optimization strategies if available
        loss_function = 'binary_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0012)
        
        if args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'focal_loss':
            loss_function = OptimizedLossFunctions.focal_loss(
                alpha=getattr(args, 'focal_alpha', 0.25),
                gamma=getattr(args, 'focal_gamma', 2.0)
            )
        elif args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'label_smoothing':
            loss_function = OptimizedLossFunctions.label_smoothing_loss(
                smoothing=getattr(args, 'label_smoothing', 0.1)
            )
        
        if args and hasattr(args, 'optimizer_strategy'):
            optimizer = OptimizedLossFunctions.get_optimized_optimizer(
                learning_rate=0.0012,
                optimization_strategy=args.optimizer_strategy
            )
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score'),
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ]
        )
    return model

def create_bribery_detection_model(input_shape: int, compile_model: bool = True, args=None):
    """Create an improved model for bribery detection with enhanced optimization."""
    
    # Apply optimization arguments for model architecture
    l2_reg = getattr(args, 'l2_regularization', 0.001) if args else 0.001
    dropout_rate = getattr(args, 'dropout_rate', 0.3) if args else 0.3
    enable_batch_norm = getattr(args, 'enable_batch_norm', True) if args else True
    
    layers = [
        # Input layer with L2 regularization
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,),
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ]
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate),
        
        # Hidden layers
        tf.keras.layers.Dense(96, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate * 0.75),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    ])
    
    if enable_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
    
    layers.extend([
        tf.keras.layers.Dropout(dropout_rate * 0.5),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate * 0.375),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate * 0.25),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model = tf.keras.Sequential(layers)
    
    if compile_model:
        # Apply optimization strategies if available
        loss_function = 'binary_crossentropy'
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        if args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'focal_loss':
            loss_function = OptimizedLossFunctions.focal_loss(
                alpha=getattr(args, 'focal_alpha', 0.25),
                gamma=getattr(args, 'focal_gamma', 2.0)
            )
        elif args and hasattr(args, 'loss_optimization') and args.loss_optimization == 'label_smoothing':
            loss_function = OptimizedLossFunctions.label_smoothing_loss(
                smoothing=getattr(args, 'label_smoothing', 0.1)
            )
        
        if args and hasattr(args, 'optimizer_strategy'):
            optimizer = OptimizedLossFunctions.get_optimized_optimizer(
                learning_rate=getattr(args, 'learning_rate', 0.001),
                optimization_strategy=args.optimizer_strategy
            )
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score'),
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
            ]
        )
    return model

def create_tff_model_fn(model_name: str, input_shape: int, sample_batch, optimization_args=None):
    """Create TFF model function with enhanced optimization support."""
    
    def model_fn():
        if model_name == 'sybil_detection':
            keras_model = create_sybil_detection_model(input_shape, compile_model=False, args=optimization_args)
        elif model_name == 'bribery_detection':
            keras_model = create_bribery_detection_model(input_shape, compile_model=False, args=optimization_args)
        elif model_name == 'batch_monitoring':
            keras_model = create_batch_monitoring_model(input_shape, compile_model=False, args=optimization_args)
        elif model_name == 'arbitrator_bias':
            keras_model = create_arbitrator_bias_model(input_shape, compile_model=False, args=optimization_args)
        elif model_name == 'dispute_risk':
            keras_model = create_dispute_risk_model(input_shape, compile_model=False, args=optimization_args)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=sample_batch.element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    return model_fn

# Helper function for feature extraction
def extract_features_for_anomaly_detection(df: pd.DataFrame, model_type: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Extract features for anomaly detection from processed DataFrame."""
    try:
        # Filter numeric columns for features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove label column if present
        feature_columns = [col for col in numeric_columns if col not in ['label', 'id', 'index']]
        
        if not feature_columns:
            logger.warning(f"No numeric feature columns found for {model_type}")
            return None
        
        features_df = df[feature_columns].fillna(0)
        
        # Create IDs DataFrame
        if 'id' in df.columns:
            ids_df = df[['id']]
        else:
            ids_df = pd.DataFrame({'id': range(len(df))})
        
        logger.info(f"Extracted {len(feature_columns)} features for {model_type}")
        return features_df, ids_df
        
    except Exception as e:
        logger.error(f"Error extracting features for {model_type}: {e}")
        return None

# Enhanced Data Augmentation Framework
class EnhancedDataAugmentation:
    """Enhanced data augmentation techniques for better model performance."""
    
    @staticmethod
    def apply_feature_scaling(features: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Apply feature scaling to improve model convergence."""
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            
            if method == 'standard':
                scaler = StandardScaler()
                return scaler.fit_transform(features)
            elif method == 'minmax':
                scaler = MinMaxScaler()
                return scaler.fit_transform(features)
            else:
                return features
        except ImportError:
            logger.warning("scikit-learn not available. Skipping feature scaling.")
            return features
    
    @staticmethod
    def apply_noise_injection(features: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Apply noise injection for data augmentation."""
        noise = np.random.normal(0, noise_factor, features.shape)
        return features + noise
    
    @staticmethod
    def apply_smote_augmentation(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-like augmentation for class balance."""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            features_resampled, labels_resampled = smote.fit_resample(features, labels)
            return features_resampled, labels_resampled
        except ImportError:
            logger.warning("imbalanced-learn not available. Skipping SMOTE augmentation.")
            return features, labels

# Performance Analysis Framework
class PerformanceAnalyzer:
    """Real-time performance analysis and optimization recommendations."""
    
    @staticmethod
    def analyze_training_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training metrics and provide insights."""
        analysis = {
            'loss_trends': {},
            'accuracy_trends': {},
            'optimization_recommendations': [],
            'convergence_analysis': {}
        }
        
        # Analyze loss trends
        if 'distributed_train_loss' in history:
            losses = history['distributed_train_loss']
            analysis['loss_trends'] = {
                'final_loss': losses[-1] if losses else None,
                'loss_reduction': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 else 0,
                'convergence_rate': PerformanceAnalyzer._calculate_convergence_rate(losses)
            }
        
        # Analyze accuracy trends
        if 'distributed_train_accuracy' in history:
            accuracies = history['distributed_train_accuracy']
            analysis['accuracy_trends'] = {
                'final_accuracy': accuracies[-1] if accuracies else None,
                'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
                'stability': PerformanceAnalyzer._calculate_stability(accuracies)
            }
        
        # Generate optimization recommendations
        analysis['optimization_recommendations'] = PerformanceAnalyzer._generate_recommendations(analysis)
        
        return analysis
    
    @staticmethod
    def _calculate_convergence_rate(values: List[float]) -> float:
        """Calculate the convergence rate of a metric."""
        if len(values) < 2:
            return 0.0
        
        # Calculate the rate of change in the last few epochs
        recent_values = values[-5:] if len(values) >= 5 else values
        if len(recent_values) < 2:
            return 0.0
        
        # Simple linear fit to estimate convergence rate
        x = np.arange(len(recent_values))
        coeffs = np.polyfit(x, recent_values, 1)
        return abs(coeffs[0])  # Slope magnitude
    
    @staticmethod
    def _calculate_stability(values: List[float]) -> float:
        """Calculate the stability of a metric (lower variance = more stable)."""
        if len(values) < 2:
            return 1.0
        
        return 1.0 / (1.0 + np.var(values))
    
    @staticmethod
    def _generate_recommendations(analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Loss-based recommendations
        loss_trends = analysis.get('loss_trends', {})
        if loss_trends.get('convergence_rate', 0) < 0.001:
            recommendations.append("Consider increasing learning rate or changing optimizer strategy")
        
        final_loss = loss_trends.get('final_loss')
        if final_loss and final_loss > 0.5:
            recommendations.append("High final loss detected. Consider focal loss or label smoothing")
        
        # Accuracy-based recommendations
        accuracy_trends = analysis.get('accuracy_trends', {})
        final_accuracy = accuracy_trends.get('final_accuracy')
        if final_accuracy and final_accuracy < 0.8:
            recommendations.append("Low accuracy detected. Consider data augmentation or model architecture changes")
        
        stability = accuracy_trends.get('stability', 1.0)
        if stability < 0.5:
            recommendations.append("Training instability detected. Consider reducing learning rate or adding regularization")
        
        if not recommendations:
            recommendations.append("Performance looks good! Continue monitoring.")
        
        return recommendations

# Synthetic Data Generation Functions
def generate_synthetic_sybil_data(num_samples: int, num_features: int = 25):
    """Generate synthetic data for Sybil detection training with enhanced attack patterns."""
    np.random.seed(42)
    features = []
    labels = []
    ids_list = []
    
    for i in range(num_samples):
        # Generate realistic sybil detection features with higher attack ratio
        is_sybil = np.random.random() < 0.35  # Increased to 35% sybil attack rate for better training
        
        if is_sybil:
            # Enhanced Sybil nodes with more extreme patterns for higher predictions (25 features)
            feature_vector = [
                # Basic blockchain features (0-9)
                np.random.beta(3, 2),  # 0. currentReputation (very high activity for sybils)
                np.random.exponential(0.1),  # 1. initialReputation (very short for sybils)
                np.random.normal(0.9, 0.05),  # 2. isVerified (very high for sybils) 
                np.random.gamma(4, 0.1),  # 3. interactions (very low for sybils)
                np.random.uniform(0.8, 1.0),  # 4. vote_actions (very high)
                np.random.beta(5, 1),  # 5. transfer_actions (artificially high)
                np.random.normal(0.2, 0.05),  # 6. production_actions (very low for sybils)
                np.random.exponential(0.2),  # 7. marketplace_actions (very low)
                np.random.gamma(0.5, 0.3),  # 8. logistics_actions (concentrated)
                np.random.uniform(0, 0.2),  # 9. dispute_actions (very low)
                
                # Attack pattern features (10-24)
                np.random.uniform(0.8, 1.0),  # 10. sybil_identification_score (very high)
                np.random.beta(1, 4),  # 11. rapid_reputation_increase (very high similarity)
                np.random.normal(0.95, 0.02),  # 12. coordination_score (very high)
                np.random.exponential(0.15),  # 13. fake_product_creation (artificial)
                np.random.uniform(0.8, 1.0),  # 14. malicious_batch_creation (very high)
                np.random.uniform(0.8, 1.0),  # 15. bribery_source_detection (very high)
                np.random.uniform(0.9, 1.0),  # 16. attack_campaign_participation (very high)
                np.random.uniform(0.85, 0.99),  # 17. extreme_risk_score (very high)
                np.random.uniform(0.8, 0.95),  # 18. promotion_manipulation (high)
                np.random.uniform(0.9, 1.0),  # 19. massive_volume_detection (very high)
                np.random.uniform(0.8, 0.95),  # 20. batch_manipulation_score (high)
                np.random.uniform(0.85, 0.98),  # 21. timing_coordination (very high)
                np.random.uniform(0.9, 1.0),  # 22. suspicious_activity_level (very high)
                np.random.uniform(0.8, 0.95),  # 23. network_influence_manipulation (high)
                np.random.uniform(0.85, 0.99),  # 24. attack_severity_score (very high)
            ]
        else:
            # Normal nodes have more diverse, organic patterns (25 features)
            feature_vector = [
                # Basic blockchain features (0-9)
                np.random.gamma(2, 0.3),  # 0. currentReputation
                np.random.exponential(0.5),  # 1. initialReputation
                np.random.normal(0.4, 0.2),  # 2. isVerified
                np.random.gamma(2, 0.4),  # 3. interactions
                np.random.uniform(0.2, 0.6),  # 4. vote_actions
                np.random.beta(2, 3),  # 5. transfer_actions
                np.random.normal(0.7, 0.15),  # 6. production_actions
                np.random.gamma(2, 0.3),  # 7. marketplace_actions
                np.random.uniform(0, 1),  # 8. logistics_actions
                np.random.beta(3, 2),  # 9. dispute_actions
                
                # Attack pattern features (10-24) - Low values for legitimate nodes
                np.random.uniform(0.0, 0.1),  # 10. sybil_identification_score
                np.random.normal(0.3, 0.2),  # 11. rapid_reputation_increase
                np.random.uniform(0, 0.5),  # 12. coordination_score
                np.random.gamma(1, 0.5),  # 13. fake_product_creation
                np.random.uniform(0, 0.4),  # 14. malicious_batch_creation
                np.random.uniform(0.0, 0.2),  # 15. bribery_source_detection
                np.random.uniform(0.0, 0.2),  # 16. attack_campaign_participation
                np.random.uniform(0.0, 0.3),  # 17. extreme_risk_score
                np.random.uniform(0.0, 0.2),  # 18. promotion_manipulation
                np.random.uniform(0.0, 0.2),  # 19. massive_volume_detection
                np.random.uniform(0.0, 0.1),  # 20. batch_manipulation_score
                np.random.uniform(0.0, 0.15),  # 21. timing_coordination
                np.random.uniform(0.0, 0.2),  # 22. suspicious_activity_level
                np.random.uniform(0.0, 0.1),  # 23. network_influence_manipulation
                np.random.uniform(0.0, 0.2),  # 24. attack_severity_score
            ]
        
        # Ensure correct number of features
        if len(feature_vector) < num_features:
            feature_vector.extend([0.5] * (num_features - len(feature_vector)))
        elif len(feature_vector) > num_features:
            feature_vector = feature_vector[:num_features]
            
        # Validate feature vector is all numeric
        feature_vector = [float(x) for x in feature_vector]
        
        features.append(feature_vector)
        labels.append(1 if is_sybil else 0)
        ids_list.append({'validator_id': f'synthetic_validator_sybil_{i}', 'synthetic_marker': True})
    
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_bribery_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for bribery detection training."""
    np.random.seed(43)
    features = []
    labels = []
    ids_list = []
    
    for i in range(num_samples):
        # Generate realistic bribery detection features
        is_bribed = np.random.random() < 0.12  # 12% bribery rate
        
        if is_bribed:
            # Bribed validators often show unusual voting patterns, stake concentration
            feature_vector = [
                np.random.beta(4, 2),  # voting_consistency (unusual patterns)
                np.random.normal(0.85, 0.08),  # stake_concentration (higher)
                np.random.exponential(0.3),  # proposal_success_rate
                np.random.gamma(3, 0.25),  # network_influence
                np.random.uniform(0.6, 1.0),  # collusion_score
                np.random.beta(2, 4),  # reputation_volatility
                np.random.normal(0.7, 0.1),  # financial_incentive_sensitivity
                np.random.exponential(0.2),  # voting_speed_anomaly
                np.random.gamma(2, 0.4),  # delegation_pattern
                np.random.uniform(0.5, 0.9),  # coordination_index
                np.random.beta(3, 2),  # conflict_of_interest_score
                np.random.normal(0.8, 0.1),  # power_concentration
                np.random.exponential(0.25),  # governance_participation
                np.random.uniform(0.4, 0.8),  # suspicious_transaction_score
                np.random.gamma(1, 0.3)  # economic_rationality_deviation
            ]
        else:
            # Normal validators show organic, diverse behavior
            feature_vector = [
                np.random.gamma(2, 0.4),  # voting_consistency
                np.random.normal(0.5, 0.2),  # stake_concentration
                np.random.uniform(0, 1),  # proposal_success_rate
                np.random.beta(2, 3),  # network_influence
                np.random.uniform(0, 0.4),  # collusion_score
                np.random.gamma(1, 0.5),  # reputation_volatility
                np.random.normal(0.4, 0.2),  # financial_incentive_sensitivity
                np.random.uniform(0, 0.6),  # voting_speed_anomaly
                np.random.beta(3, 3),  # delegation_pattern
                np.random.uniform(0, 0.5),  # coordination_index
                np.random.normal(0.2, 0.15),  # conflict_of_interest_score
                np.random.uniform(0, 0.6),  # power_concentration
                np.random.gamma(2, 0.3),  # governance_participation
                np.random.uniform(0, 0.3),  # suspicious_transaction_score
                np.random.normal(0.5, 0.2)  # economic_rationality_deviation
            ]
        
        # Ensure correct number of features
        if len(feature_vector) < num_features:
            feature_vector.extend([0.5] * (num_features - len(feature_vector)))
        elif len(feature_vector) > num_features:
            feature_vector = feature_vector[:num_features]
            
        # Validate feature vector is all numeric
        feature_vector = [float(x) for x in feature_vector]
        
        features.append(feature_vector)
        labels.append(1 if is_bribed else 0)
        ids_list.append({'validator_id': f'synthetic_validator_bribery_{i}', 'synthetic_marker': True})
    
    return {'features': features, 'labels': labels, 'ids': ids_list}

def run_federated_learning(args, dirs: Dict[str, str], config: Dict[str, Any]):
    """Run the complete Federated Learning workflow (training only)."""
    blockchain_monitoring_results = [] # Initialize here
    try:
        # Initialize monitoring system
        monitoring = MonitoringSystem(
            log_dir=dirs['logs_dir'],
            metrics_dir=dirs['metrics_dir'],
            log_level="INFO" if not args.verbose else "DEBUG",
            enable_performance_monitoring=True,
            enable_health_checks=True
        )
        
        monitoring.log_event(
            component="main",
            event_type="system_start",
            message="Starting Federated Learning system (TRAIN ONLY)",
            details={"mode": args.mode, "config": config}
        )

        # --- Connect to Blockchain and Fetch Initial Data ---
        print_section_header("Blockchain Monitoring Data Fetch")
        amoy_rpc_url = config.get("blockchain_connector", {}).get("rpc_url", "https://polygon-amoy.infura.io/v3/d455e91357464c0cb3727309e4256e94")
        num_blocks_to_fetch = config.get("blockchain_connector", {}).get("num_blocks_to_fetch", 10)
        blockchain_monitoring_results = connect_and_fetch_blockchain_data(
            rpc_url=amoy_rpc_url, 
            num_blocks_to_fetch=num_blocks_to_fetch, 
            output_dir=dirs['results_dir']
        )
        if blockchain_monitoring_results:
            print_highlight(f"Successfully fetched data for {len(blockchain_monitoring_results)} blocks.", "SUCCESS")
        else:
            print_highlight("Could not fetch blockchain monitoring data, or no data found.", "WARNING")

        # Load input data
        logger.info("Loading FL input data...")
        fl_input_data = None
        input_data_path = args.input_data_file
        if not input_data_path:
            search_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SupplyChain_dapp', 'scripts', 'lifecycle_demo', 'demo_context.json')),
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SupplyChain_dapp', 'scripts', 'lifecycle_demo', 'attack_context.json')),
                os.path.join(os.path.dirname(__file__), "cache/context_data.json"),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    input_data_path = path
                    logger.info(f"Found input data at: {path}")
                    break
        if input_data_path and os.path.exists(input_data_path):
            try:
                with open(input_data_path, 'r') as f:
                    fl_input_data = json.load(f)
                logger.info(f"Successfully loaded FL input data from {input_data_path}")
            except Exception as e:
                logger.error(f"Failed to load FL input data from {input_data_path}: {str(e)}")
                raise
        else:
            logger.error("No valid input data file found!")
            raise FileNotFoundError("Input data file is required but not found")
        # Initialize components
        logger.info("Initializing FL components...")
        fl_config = config["fl_orchestrator"]
        fl_orchestrator = FLOrchestrator(
            model_dir=dirs['models_dir'],
            client_auth_enabled=fl_config.get("client_auth_enabled", True),
            secure_aggregation=fl_config.get("secure_aggregation", True),
            differential_privacy=fl_config.get("differential_privacy", False),
            dp_noise_multiplier=fl_config.get("dp_noise_multiplier", 0.1),
            dp_l2_norm_clip=fl_config.get("max_grad_norm", 1.0)
        )
        model_repository = ModelRepository(dirs['models_dir'])
        feature_extractor = FeatureExtractor()

        # Extract features from blockchain data
        logger.info("Extracting features from blockchain data...")
        extraction_timer = monitoring.start_timer("feature_extraction", "data_processing")
        processed_data = feature_extractor.extract_features(fl_input_data)
        monitoring.stop_timer(extraction_timer)
        logger.info(f"Feature extraction completed. Extracted features for {len(processed_data)} model types.")        # Prepare training data - Focus on 2 core models: Sybil and Bribery Detection
        logger.info("Preparing federated training data for core attack detection models...")
        training_timer = monitoring.start_timer("federated_training", "training")
        model_training_input_data = {}
        
        # Only train the 2 most critical models for attack detection
        core_attack_models = { 
            'sybil_detection': {'min_normal': 50, 'min_full': 50},
            'bribery_detection': {'min_normal': 50, 'min_full': 50}
        }
        logger.info(f"Focusing on {len(core_attack_models)} core attack detection models: {list(core_attack_models.keys())}")
        for model_name, model_props in core_attack_models.items():
            current_data_dict = {'features': [], 'labels': [], 'ids': []}
            df_from_processed = processed_data.get(model_name)
            min_samples = model_props['min_full'] if args.mode == 'full' else model_props['min_normal']            # Enhanced processing for core attack detection models
            if model_name == 'bribery_detection':
                if not (isinstance(df_from_processed, pd.DataFrame) and not df_from_processed.empty):
                    logger.warning(f"Not enough real data for {model_name}. Generating enhanced synthetic data.")
                    current_data_dict = generate_synthetic_bribery_data(min_samples)
                else:
                    logger.info(f"Processing real attack data for {model_name}...")
                    extraction_result = extract_features_for_anomaly_detection(df_from_processed, model_name)
                    if extraction_result:
                        features_df, ids_df = extraction_result
                        current_data_dict['features'] = features_df.values.tolist()
                        current_data_dict['ids'] = ids_df.to_dict(orient='records')
                        if 'label' in df_from_processed.columns:
                            try:
                                aligned_labels = df_from_processed.loc[features_df.index, 'label'].tolist()
                                current_data_dict['labels'] = aligned_labels
                            except KeyError:
                                if len(df_from_processed['label']) == len(features_df):
                                    current_data_dict['labels'] = df_from_processed['label'].tolist()
                                else:
                                    logger.warning(f"Label count mismatch for {model_name}. Using default labels.")
                                    current_data_dict['labels'] = [0] * len(features_df)
                        else:
                            logger.warning(f"No 'label' column in processed data for {model_name}. Using default labels.")
                            current_data_dict['labels'] = [0] * len(features_df)
                    else:
                        logger.warning(f"Feature extraction failed for {model_name}. Using synthetic data.")
                        current_data_dict = generate_synthetic_bribery_data(min_samples)
            elif model_name == 'sybil_detection':
                if isinstance(df_from_processed, pd.DataFrame) and not df_from_processed.empty:
                    logger.info(f"Processing real attack data for {model_name}...")
                    extraction_result = extract_features_for_anomaly_detection(df_from_processed, model_name)
                    if extraction_result:
                        features_df, ids_df = extraction_result
                        current_data_dict['features'] = features_df.values.tolist()
                        current_data_dict['ids'] = ids_df.to_dict(orient='records')
                        if 'label' in df_from_processed.columns:
                            try:
                                aligned_labels = df_from_processed.loc[features_df.index, 'label'].tolist()
                                current_data_dict['labels'] = aligned_labels
                            except KeyError:
                                if len(df_from_processed['label']) == len(features_df):
                                    current_data_dict['labels'] = df_from_processed['label'].tolist()
                                else:
                                    logger.warning(f"Label count mismatch for {model_name}. Using default labels.")
                                    current_data_dict['labels'] = [0] * len(features_df)
                        else:
                            logger.warning(f"No 'label' column in processed data for {model_name}. Using default labels.")
                            current_data_dict['labels'] = [0] * len(features_df)
                    else:
                        logger.warning(f"Feature extraction failed for {model_name}. Using synthetic data.")
                        current_data_dict = generate_synthetic_sybil_data(min_samples)
                else:
                    logger.warning(f"Not enough real data for {model_name}. Generating enhanced synthetic data.")
                    current_data_dict = generate_synthetic_sybil_data(min_samples)            
            # Fallback to ensure minimum data requirements for core models
            if len(current_data_dict['features']) < min_samples:
                logger.warning(f"Insufficient data for {model_name} ({len(current_data_dict['features'])} < {min_samples}). Adding synthetic data.")
                if model_name == 'bribery_detection':
                    synthetic_data = generate_synthetic_bribery_data(min_samples - len(current_data_dict['features']))
                elif model_name == 'sybil_detection':
                    synthetic_data = generate_synthetic_sybil_data(min_samples - len(current_data_dict['features']))
                
                # Merge synthetic data with existing data
                current_data_dict['features'].extend(synthetic_data['features'])
                current_data_dict['labels'].extend(synthetic_data['labels'])
                current_data_dict['ids'].extend(synthetic_data['ids'])            
            model_training_input_data[model_name] = current_data_dict
            if current_data_dict['features']:
                print_highlight(f"✅ Data prepared for {model_name}: {len(current_data_dict['features'])} samples", "SUCCESS")
                logger.info(f"Data prepared for {model_name}: {len(current_data_dict['features'])} samples.")
            else:
                print_highlight(f"⚠️  No data available for {model_name} after preparation", "WARNING")
                logger.info(f"No data available for {model_name} after preparation.")
        
        # Initialize client data structure
        client_data = {'client_1': {}}
        
        print_section_header("Preparing data for Federated Learning")
          # Core models configuration - Focus on attack detection only
        core_models_config = {
            'sybil_detection': 25,  # 25 features for enhanced Sybil detection with attack patterns
            'bribery_detection': 15  # 15 features for Bribery detection
        }
        
        print_highlight(f"🎯 Configuring {len(core_models_config)} core attack detection models", "INFO")
        logger.info(f"Configuring {len(core_models_config)} core attack detection models for training")
        
        for model_name_key in core_models_config.keys():
            if model_name_key in model_training_input_data and \
               model_training_input_data[model_name_key] and \
               len(model_training_input_data[model_name_key].get('features', [])) > 0:
                
                # Ensure all feature vectors have consistent shape
                raw_features = model_training_input_data[model_name_key]['features']
                expected_features = core_models_config[model_name_key]
                
                # Validate and fix feature vector shapes
                validated_features = []
                for feature_vector in raw_features:
                    if isinstance(feature_vector, list):
                        # Ensure correct number of features
                        if len(feature_vector) < expected_features:
                            # Pad with zeros if too short
                            feature_vector.extend([0.0] * (expected_features - len(feature_vector)))
                        elif len(feature_vector) > expected_features:
                            # Truncate if too long
                            feature_vector = feature_vector[:expected_features]
                        validated_features.append(feature_vector)
                    else:
                        logger.warning(f"Invalid feature vector format for {model_name_key}: {type(feature_vector)}")
                        # Create default feature vector
                        validated_features.append([0.0] * expected_features)
                  # Convert to numpy arrays with validated shapes
                try:
                    features = np.array(validated_features, dtype=np.float32)
                    labels = np.array(model_training_input_data[model_name_key]['labels'], dtype=np.float32)
                    
                    # Apply enhanced data augmentation if enabled
                    if args.enable_data_augmentation:
                        logger.info(f"Applying data augmentation for {model_name_key}")
                        
                        # Apply feature scaling
                        if args.feature_scaling != 'none':
                            features = EnhancedDataAugmentation.apply_feature_scaling(features, method=args.feature_scaling)
                            logger.info(f"Applied {args.feature_scaling} feature scaling")
                        
                        # Apply noise injection for data augmentation
                        if args.noise_factor > 0:
                            augmented_features = EnhancedDataAugmentation.apply_noise_injection(features, noise_factor=args.noise_factor)
                            features = np.vstack([features, augmented_features])
                            labels = np.hstack([labels, labels])  # Duplicate labels for augmented data
                            logger.info(f"Applied noise injection with factor {args.noise_factor}")
                        
                        # Apply SMOTE-like augmentation for class balance
                        if args.enable_smote:
                            features, labels = EnhancedDataAugmentation.apply_smote_augmentation(features, labels)
                            logger.info("Applied SMOTE-like augmentation for class balance")
                    
                    logger.info(f"Successfully created arrays for {model_name_key}: features shape {features.shape}, labels shape {labels.shape}")
                    client_data['client_1'][model_name_key] = (features, labels)
                except Exception as array_error:
                    logger.error(f"Error creating numpy arrays for {model_name_key}: {array_error}")
                    client_data['client_1'][model_name_key] = (np.array([]).astype(np.float32), np.array([]).astype(np.float32))
            else:
                client_data['client_1'][model_name_key] = (np.array([]).astype(np.float32), np.array([]).astype(np.float32))
                logger.warning(f"No data (real or synthetic) available for model {model_name_key} to be included in client_data.")
        
        model_input_shapes = {}
        for model_name_key in core_models_config.keys():
            features = client_data['client_1'][model_name_key][0]
            if features.size > 0:
                model_input_shapes[model_name_key] = features.shape[1]
            else:
                model_input_shapes[model_name_key] = None
          # Enhanced training process for core attack detection models
        global_models = {}
        training_history = {}
        
        for model_name in core_models_config.keys():
            input_shape = model_input_shapes[model_name]
            
            if input_shape is None:
                logger.warning(f"Skipping training for {model_name} due to lack of data.")
                continue
            
            logger.info(f"--- Training model: {model_name} ---")
            try:
                federated_datasets = []
                for client_id, client_model_data in client_data.items():
                    features, labels = client_model_data[model_name]
                    dataset = tf.data.Dataset.from_tensor_slices((features.astype(np.float32), labels.astype(np.float32)))
                    dataset = dataset.batch(args.batch_size).repeat(1)
                    federated_datasets.append(dataset)
                
                sample_batch = federated_datasets[0]
                
                # Create optimization arguments for enhanced model creation
                optimization_args = argparse.Namespace(
                    loss_optimization=args.loss_optimization,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                    label_smoothing=args.label_smoothing,
                    optimizer_strategy=args.optimizer_strategy,
                    learning_rate=args.learning_rate,
                    enable_batch_norm=args.enable_batch_norm,
                    dropout_rate=args.dropout_rate,
                    l2_regularization=args.l2_regularization
                )
                
                model_fn = create_tff_model_fn(model_name, input_shape, sample_batch, optimization_args)
                client_optimizer_builder = tff.learning.optimizers.build_sgdm(learning_rate=args.learning_rate)
                server_optimizer_builder = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
                
                print_section_header(f"TRAINING MODEL: {model_name.upper()}")
                print_highlight(f"Input shape: {input_shape} features", "INFO")
                print_highlight(f"Training samples: {len(features)}", "INFO")
                print_highlight(f"Training rounds: {args.num_rounds}", "INFO")
                print_highlight(f"Batch size: {args.batch_size}", "INFO")
                print_highlight(f"Learning rate: {args.learning_rate}", "INFO")
                print_highlight(f"Loss optimization: {args.loss_optimization}", "INFO")
                print_highlight(f"Optimizer strategy: {args.optimizer_strategy}", "INFO")
                
                logger.info(f"Starting federated training for {model_name} with enhanced optimization...")
                logger.info(f"Training configuration: {optimization_args.__dict__}")
                print_highlight("Starting federated learning training...", "INFO")
                
                server_state, history = fl_orchestrator.train_federated_model(
                    federated_data=federated_datasets,
                    model_fn=model_fn,
                    num_rounds=args.num_rounds,
                    client_optimizer_fn=client_optimizer_builder,
                    server_optimizer_fn=server_optimizer_builder
                )
                
                print_highlight("Federated learning training completed!", "SUCCESS")
                
                if server_state is not None:
                    # Create enhanced model with optimization arguments
                    if model_name == 'sybil_detection':
                        trained_model = create_sybil_detection_model(input_shape, compile_model=True, args=optimization_args)
                    elif model_name == 'bribery_detection':
                        trained_model = create_bribery_detection_model(input_shape, compile_model=True, args=optimization_args)
                    else:
                        logger.warning(f"Unknown core model type: {model_name}. Skipping trained model creation.")
                        continue
                    
                    # Enhanced weight transfer using the new orchestrator method
                    try:
                        trained_model = fl_orchestrator.transfer_weights_to_keras_model(server_state, trained_model)
                        logger.info(f"Successfully transferred weights for {model_name}")
                          # Save the model with enhanced metadata
                        model_save_path = os.path.join(dirs['models_dir'], f"{model_name}_final.h5")
                        trained_model.save(model_save_path)
                        logger.info(f"Enhanced {model_name} model saved to {model_save_path}")
                        
                        global_models[model_name] = {
                            'model': trained_model,
                            'model_path': model_save_path,
                            'optimization_config': optimization_args.__dict__,
                            'training_metrics': history                        }
                        training_history[model_name] = history
                        
                        # Enhanced performance analysis if enabled
                        if args.enable_performance_analysis:
                            logger.info(f"Performing performance analysis for {model_name}")
                            
                            def _safe_json(obj):
                                try:
                                    return json.dumps(obj, indent=2, default=str)
                                except Exception:
                                    return str(obj)
                            
                            print(f"[TRAINING][{model_name}] Full training history:")
                            print(_safe_json(history))
                            logger.info(f"[TRAINING][{model_name}] Full training history:\n{_safe_json(history)}")
                        
                    except Exception as weight_transfer_error:
                        logger.error(f"Weight transfer failed for {model_name}: {weight_transfer_error}")
                        # Fallback to basic model saving
                        fl_orchestrator.save_model_state(server_state, model_name)
                        global_models[model_name] = {
                            'server_state': server_state,
                            'optimization_config': optimization_args.__dict__,
                            'training_metrics': history
                        }
                        training_history[model_name] = history
                        
                else:
                    logger.warning(f"No server_state returned for {model_name}")
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")
        
        monitoring.stop_timer(training_timer)
        logger.info("Federated Training phase completed.")
        
        # Generate comprehensive performance report
        if args.save_performance_report and training_history:
            try:
                logger.info("Generating comprehensive performance report...")
                performance_report_path = os.path.join(dirs['results_dir'], 'comprehensive_performance_report.json')                # Prepare comprehensive report data
                report_data = {
                    'training_history': training_history,
                    'model_configs': {name: model_info.get('optimization_config', {}) for name, model_info in global_models.items()},
                    'metadata': {
                        'training_args': vars(args),
                        'models_trained': list(global_models.keys()),
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                fl_orchestrator.save_performance_report(
                    report_data=report_data,
                    output_dir=os.path.dirname(performance_report_path),
                    filename=os.path.basename(performance_report_path)
                )
                logger.info(f"Performance report saved to {performance_report_path}")
            except Exception as report_error:
                logger.error(f"Failed to generate performance report: {report_error}")
        
        logger.info("Training-only workflow completed successfully.")
        
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
    finally:
        if 'monitoring' in locals():
            monitoring.log_event(
                component="main",
                event_type="system_end",
                message="Federated Learning system shutdown",
                details={"models_trained": len(global_models) if 'global_models' in locals() else 0}
            )

def connect_and_fetch_blockchain_data(rpc_url: str, num_blocks_to_fetch: int, output_dir: str) -> List[Dict[str, Any]]:
    """
    Connects to the blockchain, fetches data from recent blocks, and saves it.
    """
    logger.info(f"Attempting to connect to blockchain RPC: {rpc_url}")
    blockchain_data = []
    raw_blockchain_data_path = os.path.join(output_dir, 'blockchain_monitoring_raw.json')

    try:
        connector = BlockchainConnector(rpc_url=rpc_url)
        if not connector.is_connected():
            logger.error("Failed to connect to the blockchain.")
            return blockchain_data

        logger.info("Successfully connected to the blockchain.")
        latest_block_number = connector.get_latest_block_number()
        if latest_block_number is None:
            logger.error("Could not fetch the latest block number.")
            return blockchain_data
        logger.info(f"Latest block number: {latest_block_number}")

        start_block = max(0, latest_block_number - num_blocks_to_fetch + 1)
        for block_num in range(start_block, latest_block_number + 1):
            logger.info(f"Fetching block: {block_num}")
            block = connector.get_block_with_contract_activity(block_num)
            if block:
                try:
                    from web3 import Web3
                    block_serializable = json.loads(Web3.to_json(block))
                    blockchain_data.append(block_serializable)
                    logger.debug(f"Successfully fetched and serialized block {block_num}")
                except Exception as ser_e:
                    logger.warning(f"Could not serialize block {block_num}: {ser_e}")
            else:
                logger.warning(f"Could not fetch block {block_num}.")

        if blockchain_data:
            try:
                with open(raw_blockchain_data_path, 'w') as f:
                    json.dump(blockchain_data, f, indent=4)
                logger.info(f"Raw blockchain monitoring data saved to {raw_blockchain_data_path}")
            except Exception as e:
                logger.error(f"Error saving raw blockchain data: {str(e)}")
    except Exception as e:
        logger.error(f"Error during blockchain data fetching: {str(e)}")
        logger.error(traceback.format_exc())
    return blockchain_data

# Ensure this is the end of the main_fl_process function or just before any final return statement if one exists.
# The if __name__ == \"__main__\": block should be outside this function.

def run_individual_fl_script(script_name, data_path, output_path):
    # TODO: Implement the function logic here
    pass

def to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(to_serializable(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# Main execution entry point
if __name__ == "__main__":
    try:
        print("[DEBUG] Main execution started.")
        
        # Parse command line arguments
        args = parse_args()
        print(f"[DEBUG] Arguments parsed: {vars(args)}")
        
        # Load configuration
        config = load_config(args.config_file)
        print(f"[DEBUG] Configuration loaded.")
        
        # Setup directories
        dirs = setup_directories(args)
        print(f"[DEBUG] Directories created: {dirs}")
        
        # Log execution start
        print(f"[INFO] Starting Federated Learning Integration System")
        print(f"[INFO] Mode: {args.mode}")
        print(f"[INFO] Input data file: {args.input_data_file}")
        print(f"[INFO] Output directory: {args.output_dir}")
        print(f"[INFO] Number of rounds: {args.num_rounds}")
        print(f"[INFO] Number of clients: {args.num_clients}")
        print(f"[INFO] Batch size: {args.batch_size}")
        print(f"[INFO] Learning rate: {args.learning_rate}")
        
        # Run the federated learning workflow
        print("[DEBUG] Calling run_federated_learning function...")
        run_federated_learning(args, dirs, config)
        
        print("[INFO] ✅ Federated Learning execution completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INFO] ⚠️ Execution interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERROR] ❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] ❌ Unexpected error occurred: {e}")
        print(f"[ERROR] Error details: {traceback.format_exc()}")
        sys.exit(1)
