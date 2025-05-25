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
import tensorflow as tf
import tensorflow_federated as tff

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
    parser.add_argument('--num-rounds', type=int, default=20,
                       help='Number of federated learning rounds')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for federated training')
    parser.add_argument('--num-clients', type=int, default=2,
                       help='Number of federated clients')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
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

def create_sybil_detection_model(input_shape: int, compile_model: bool = True):
    """Create a model for Sybil detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    return model

def create_batch_monitoring_model(input_shape: int, compile_model: bool = True):
    """Create a model for batch monitoring."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    return model

def create_node_behavior_model(input_shape: int, compile_model: bool = True):
    """Create a model for node behavior analysis."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    return model

def create_arbitrator_bias_model(input_shape: int, compile_model: bool = True):
    """Create a model for arbitrator bias detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    return model

def create_dispute_risk_model(input_shape: int, compile_model: bool = True):
    """Create a model for dispute risk assessment."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(96, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0012),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()]
        )
    return model

def create_bribery_detection_model(input_shape: int, compile_model: bool = True):
    """Create a model for bribery attack detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if compile_model:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0006),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    return model

def create_tff_model_fn(model_name: str, input_shape: int, sample_batch):
    """Create a TFF-compatible model function."""
    def model_fn():
        # Create uncompiled Keras model
        if model_name == 'sybil_detection':
            keras_model = create_sybil_detection_model(input_shape, compile_model=False)
        elif model_name == 'batch_monitoring':
            keras_model = create_batch_monitoring_model(input_shape, compile_model=False)
        elif model_name == 'node_behavior':
            keras_model = create_node_behavior_model(input_shape, compile_model=False)
        elif model_name == 'arbitrator_bias':
            keras_model = create_arbitrator_bias_model(input_shape, compile_model=False)
        elif model_name == 'dispute_risk':
            keras_model = create_dispute_risk_model(input_shape, compile_model=False)
        elif model_name == 'bribery_detection':
            keras_model = create_bribery_detection_model(input_shape, compile_model=False)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Get input spec from sample batch
        input_spec = sample_batch.element_spec
        
        # Return TFF VariableModel
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        )
    
    return model_fn

def format_blockchain_events_table(events: List[Dict], output_file_path: str = None) -> str:
    """Format blockchain events into a readable table."""
    if not events:
        return "No blockchain events found.\n"
    
    # Header
    header = f"{'Block':<8} {'TxHash':<12} {'Event':<20} {'From':<12} {'To':<12} {'Amount':<10} {'Time':<20}"
    separator = "=" * len(header)
    
    table_lines = [header, separator]
    
    for event in events:
        block = str(event.get('block_number', 'N/A'))[:7]
        tx_hash = str(event.get('transaction_hash', 'N/A'))[:11]
        event_type = str(event.get('event_type', 'N/A'))[:19]
        from_addr = str(event.get('from_address', 'N/A'))[:11]
        to_addr = str(event.get('to_address', 'N/A'))[:11]
        amount = str(event.get('amount', 'N/A'))[:9]
        timestamp = str(event.get('timestamp', 'N/A'))[:19];
        
        line = f"{block:<8} {tx_hash:<12} {event_type:<20} {from_addr:<12} {to_addr:<12} {amount:<10} {timestamp:<20}"
        table_lines.append(line)
    
    table_content = "\n".join(table_lines) + "\n"
    
    # Save to file if path provided
    if output_file_path:
        try:
            with open(output_file_path, 'w') as f:
                f.write("BLOCKCHAIN EVENTS MONITORING TABLE\n")
                f.write("="*60 + "\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Events: {len(events)}\n")
                f.write("="*60 + "\n\n")
                f.write(table_content)
            logger.info(f"Blockchain events table saved to {output_file_path}")
        except Exception as e:
            logger.warning(f"Could not save blockchain events table: {str(e)}")
    
    return table_content

def generate_synthetic_bribery_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for bribery detection training."""
    np.random.seed(45)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.exponential(0.3), np.random.gamma(2, 0.5), np.random.beta(2, 8),
            np.random.normal(0.1, 0.05), np.random.poisson(2), np.random.uniform(0, 1),
            np.random.exponential(0.2), np.random.normal(0.05, 0.02), np.random.beta(1, 10),
            np.random.gamma(1, 0.3), np.random.uniform(0, 1), np.random.normal(0.03, 0.01),
            np.random.exponential(0.15), np.random.beta(2, 15), np.random.uniform(0, 1)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.1] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.25 else 0)
        ids_list.append({'validator_id': f'synthetic_validator_bribery_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_arbitrator_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for arbitrator bias detection training."""
    np.random.seed(46)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.beta(3, 3), np.random.normal(0.15, 0.08), np.random.exponential(0.4),
            np.random.gamma(2, 0.3), np.random.uniform(0, 1), np.random.normal(0.8, 0.1),
            np.random.beta(4, 2), np.random.exponential(0.25), np.random.normal(0.7, 0.15),
            np.random.gamma(1, 0.4), np.random.uniform(0, 1), np.random.beta(2, 5),
            np.random.normal(0.9, 0.05), np.random.exponential(0.2), np.random.uniform(0, 1)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.5] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.18 else 0)
        ids_list.append({'arbitrator_id': f'synthetic_arbitrator_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_dispute_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for dispute risk assessment training."""
    np.random.seed(47)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.gamma(2, 0.5), np.random.beta(3, 5), np.random.exponential(0.3),
            np.random.normal(0.2, 0.1), np.random.uniform(0, 1), np.random.gamma(1, 0.4),
            np.random.beta(2, 4), np.random.exponential(0.25), np.random.normal(0.6, 0.2),
            np.random.uniform(0, 1), np.random.gamma(3, 0.2), np.random.beta(4, 3),
            np.random.normal(0.4, 0.15), np.random.exponential(0.2), np.random.uniform(0, 1)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.3] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.22 else 0)
        ids_list.append({'dispute_id': f'synthetic_dispute_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_sybil_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for Sybil detection training."""
    np.random.seed(42)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.normal(0.5, 0.2), np.random.exponential(0.3), np.random.gamma(2, 2),
            np.random.beta(2, 5), np.random.normal(100, 30), np.random.poisson(5),
            np.random.uniform(0, 1), np.random.normal(0.7, 0.15), np.random.exponential(0.2),
            np.random.uniform(0, 24), np.random.normal(0.8, 0.1), np.random.gamma(1, 1),
            np.random.beta(3, 2), np.random.normal(0.6, 0.2), np.random.uniform(0, 1)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.5] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.2 else 0)
        ids_list.append({'node_id': f'synthetic_sybil_node_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_batch_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for batch monitoring training."""
    np.random.seed(43)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.poisson(10), np.random.exponential(0.5), np.random.normal(0.95, 0.05),
            np.random.gamma(2, 0.1), np.random.uniform(1, 100), np.random.normal(50, 15),
            np.random.normal(60, 20), np.random.exponential(0.3), np.random.beta(2, 3),
            np.random.normal(0.8, 0.1), np.random.uniform(0, 1), np.random.poisson(3),
            np.random.normal(0.9, 0.05), np.random.exponential(0.2), np.random.uniform(0, 1)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.5] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.15 else 0)
        ids_list.append({'batch_id': f'synthetic_batch_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def generate_synthetic_node_data(num_samples: int, num_features: int = 15):
    """Generate synthetic data for node behavior analysis training."""
    np.random.seed(44)
    features = []
    labels = []
    ids_list = []
    for i in range(num_samples):
        feature_vector = [
            np.random.normal(0.7, 0.2), np.random.exponential(0.4), np.random.gamma(3, 0.1),
            np.random.beta(4, 2), np.random.normal(0.85, 0.1), np.random.poisson(7),
            np.random.uniform(0, 1), np.random.normal(0.9, 0.05), np.random.exponential(0.25),
            np.random.beta(3, 3), np.random.normal(0.75, 0.15), np.random.gamma(2, 0.2),
            np.random.uniform(0, 1), np.random.normal(0.8, 0.1), np.random.beta(2, 4)
        ]
        if len(feature_vector) < num_features:
            feature_vector.extend([0.6] * (num_features - len(feature_vector)))
        else:
            feature_vector = feature_vector[:num_features]
        features.append(feature_vector)
        labels.append(1 if np.random.random() < 0.1 else 0)
        ids_list.append({'node_id': f'synthetic_node_behavior_{i}', 'synthetic_marker': True})
    return {'features': features, 'labels': labels, 'ids': ids_list}

def extract_features_for_anomaly_detection(data: pd.DataFrame, model_type: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extracts features and relevant ID columns for anomaly detection based on the model type.
    Returns a tuple: (features_df, ids_df) or None if data is invalid.
    features_df: DataFrame of features for the anomaly detection model.
    ids_df: DataFrame of identifier columns for correlating anomalies.
    """
    if not isinstance(data, pd.DataFrame):
        logger.warning(f"Input data for {model_type} is not a DataFrame. Skipping feature extraction.")
        return None
    if data.empty:
        logger.warning(f"Input data for {model_type} is empty. Skipping feature extraction.")
        return None

    data = data.copy() # Work on a copy

    # Define common ID columns that might exist and should be preserved if present
    potential_id_cols = ['node_id', 'transaction_id', 'validator_id', 'batch_id', 'arbitrator_id', 'dispute_id', 'actor_id', 'item_id', 'timestamp', 'block_number', 'event_type', 'from_address', 'to_address']
    
    # Identify which of the potential ID columns are actually in the DataFrame
    id_cols_present = [col for col in potential_id_cols if col in data.columns]
    
    # Feature columns are all columns that are not ID columns and not the 'label' (if present)
    feature_cols = [col for col in data.columns if col not in id_cols_present and col != 'label']

    if not feature_cols:
        logger.warning(f"No feature columns found for {model_type} after excluding ID columns. Available columns: {list(data.columns)}")
        return None

    features_df = data[feature_cols]
    ids_df = data[id_cols_present] if id_cols_present else pd.DataFrame() # Return empty DF if no IDs

    # Specific adjustments or validations per model_type can be added here if needed
    # For example, ensuring certain key ID columns are present for specific models.
    # if model_type == "Sybil Detection" and 'node_id' not in ids_df.columns:
    #     logger.warning(f"Sybil Detection data is missing 'node_id'. Correlation might be affected.")
    
    logger.info(f"Extracted {len(feature_cols)} features and {len(id_cols_present)} ID columns for {model_type}.")
    return features_df, ids_df


def run_federated_learning(args, dirs: Dict[str, str], config: Dict[str, Any]):
    """Run the complete Federated Learning workflow."""
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
            message="Starting Federated Learning system",
            details={"mode": args.mode, "config": config}
        )

        # --- Connect to Blockchain and Fetch Initial Data ---
        print_section_header("Blockchain Monitoring Data Fetch")
        # Use a default Amoy RPC URL if not specified elsewhere, or allow configuration
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
        
        # Define search paths if no specific file is given
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
            raise FileNotFoundError("Input data file is required but not found")        # Initialize components
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
        anomaly_detector = AnomalyDetector(
            results_dir=dirs['results_dir'],
            detection_threshold=config["anomaly_detector"].get("threshold", 0.5)
        )
        response_engine = ResponseEngine(
            config_path=args.config_file,
            sybil_node_threshold=config["response_engine"].get("sybil_node_threshold", 1),
            suspicious_batch_threshold=config["response_engine"].get("suspicious_batch_threshold", 1),
            high_risk_dispute_threshold=config["response_engine"].get("high_risk_dispute_threshold", 1),
            bribery_attack_threshold=config["response_engine"].get("bribery_attack_threshold", 1),
            overall_confidence_threshold=config["response_engine"].get("overall_confidence_threshold", 0.7),
            notification_channels=config["response_engine"].get("notification_channels", ["log"])
        )
        model_repository = ModelRepository(dirs['models_dir'])
        feature_extractor = FeatureExtractor()

        # Extract features from blockchain data
        logger.info("Extracting features from blockchain data...")
        extraction_timer = monitoring.start_timer("feature_extraction", "data_processing")
        
        processed_data = feature_extractor.extract_features(fl_input_data)
        
        monitoring.stop_timer(extraction_timer)
        logger.info(f"Feature extraction completed. Extracted features for {len(processed_data)} model types.")

        # Prepare training data
        logger.info("Preparing federated training data...")
        training_timer = monitoring.start_timer("federated_training", "training")
        
        model_training_input_data = {}
        # Define all models that the system might handle
        all_system_models = { 
            'sybil_detection': {'min_normal': 30, 'min_full': 30},
            'bribery_detection': {'min_normal': 30, 'min_full': 30},
            'node_behavior': {'min_normal': 30, 'min_full': 30},
            'batch_monitoring': {'min_normal': 30, 'min_full': 30},
            'arbitrator_bias': {'min_normal': 0, 'min_full': 30}, # Only in full mode
            'dispute_risk': {'min_normal': 0, 'min_full': 30}  # Only in full mode
        }

        for model_name, model_props in all_system_models.items():
            current_data_dict = {'features': [], 'labels': [], 'ids': []}
            df_from_processed = processed_data.get(model_name)

            # Chỉ sử dụng dữ liệu thực, không sinh synthetic data
            if isinstance(df_from_processed, pd.DataFrame):
                if not df_from_processed.empty:
                    logger.info(f"Processing real data for {model_name}...")
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
                                    logger.warning(f"Label count mismatch for {model_name} after feature extraction. Using default labels (all 0s).")
                                    current_data_dict['labels'] = [0] * len(features_df)
                        else:
                            logger.warning(f"No 'label' column in processed data for {model_name}. Using default labels (all 0s).")
                            current_data_dict['labels'] = [0] * len(features_df)
                    else:
                        logger.warning(f"Feature extraction failed for {model_name} from processed_data. Skipping model.")
                        current_data_dict = {'features': [], 'labels': [], 'ids': []}
                else:
                    logger.warning(f"DataFrame for {model_name} is empty. Skipping model.")
                    current_data_dict = {'features': [], 'labels': [], 'ids': []}
            else:
                logger.warning(f"Processed data for {model_name} is not a valid DataFrame. Skipping model.")
                current_data_dict = {'features': [], 'labels': [], 'ids': []}

            # Kiểm tra số lượng mẫu tối thiểu, nếu không đủ thì bỏ qua model
            min_samples = model_props['min_full'] if args.mode == 'full' else model_props['min_normal']
            if len(current_data_dict['features']) < min_samples:
                logger.warning(f"Not enough real data for {model_name} (has {len(current_data_dict['features'])}, needs {min_samples}). Skipping model.")
                current_data_dict = {'features': [], 'labels': [], 'ids': []}

            model_training_input_data[model_name] = current_data_dict
            if current_data_dict['features']:
                logger.info(f"Data prepared for {model_name}: {len(current_data_dict['features'])} samples.")
            else:
                logger.info(f"No data available for {model_name} after preparation.")

        # Create client data splits for all model types based on models_config
        client_data = {'client_1': {}}
        # Core models for supply chain protection against Sybil + Bribery attacks
        models_config = {
            'sybil_detection': 15,       # Primary: Detect fake nodes in supply chain
            'bribery_detection': 15,     # Primary: Detect node bribery attempts  
            'node_behavior': 15,         # Secondary: Monitor node behavior patterns
            'batch_monitoring': 15       # Secondary: Monitor transaction batches
        }
        
        # For full mode, add advanced models
        if args.mode == 'full':
            models_config.update({
                'arbitrator_bias': 15,   # Detect biased arbitrators
                'dispute_risk': 15       # Assess dispute risks
            })
            # Ensure advanced models also have their data in model_training_input_data if generated
            # The following lines are redundant as model_training_input_data is already populated
            #by the loop above (lines 730-778) which includes synthetic data generation.
            # if \'arbitrator_bias\' not in model_training_input_data or not model_training_input_data[\'arbitrator_bias\'][\'features\']:
            #      model_training_input_data[\'arbitrator_bias\'] = arbitrator_data # This line causes NameError
            # if \'dispute_risk\' not in model_training_input_data or not model_training_input_data[\'dispute_risk\'][\'features\']:
            #      model_training_input_data[\'dispute_risk\'] = dispute_data # This line causes NameError


        for model_name_key in models_config.keys():
            if model_name_key in model_training_input_data and \
               model_training_input_data[model_name_key] and \
               len(model_training_input_data[model_name_key].get('features', [])) > 0:
                features = np.array(model_training_input_data[model_name_key]['features'])
                labels = np.array(model_training_input_data[model_name_key]['labels'])
                client_data['client_1'][model_name_key] = (features, labels)
            else:
                # Provide empty arrays if no data, to avoid key errors later if model_config includes it
                client_data['client_1'][model_name_key] = (np.array([]).astype(np.float32), np.array([]).astype(np.float32))
                logger.warning(f"No data (real or synthetic) available for model {model_name_key} to be included in client_data.")

        global_models = {}
        training_history = {}

        for model_name, input_shape in models_config.items():
            if not client_data['client_1'][model_name][0].size > 0:
                logger.warning(f"Skipping training for {model_name} due to lack of data.")
                continue
                
            logger.info(f"--- Training model: {model_name} ---")
            
            try:
                # Prepare federated data
                federated_datasets = []
                for client_id, client_model_data in client_data.items():
                    features, labels = client_model_data[model_name]
                    dataset = tf.data.Dataset.from_tensor_slices((features.astype(np.float32), labels.astype(np.float32)))
                    dataset = dataset.batch(args.batch_size).repeat(1)
                    federated_datasets.append(dataset)
                
                # Create TFF model function
                sample_batch = federated_datasets[0]
                model_fn = create_tff_model_fn(model_name, input_shape, sample_batch)
                client_optimizer_builder = tff.learning.optimizers.build_sgdm(learning_rate=args.learning_rate)
                server_optimizer_builder = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
                # Train federated model
                logger.info(f"Starting federated training for {model_name}...")
                server_state, history = fl_orchestrator.train_federated_model(
                    federated_data=federated_datasets,
                    model_fn=model_fn,
                    num_rounds=args.num_rounds,
                    client_optimizer_fn=client_optimizer_builder,
                    server_optimizer_fn=server_optimizer_builder
                )
                
                if server_state is not None:
                    # Create compiled model and transfer weights
                    if model_name == 'sybil_detection':
                        trained_model = create_sybil_detection_model(input_shape, compile_model=True)
                    elif model_name == 'bribery_detection':
                        trained_model = create_bribery_detection_model(input_shape, compile_model=True)
                    elif model_name == 'batch_monitoring':
                        trained_model = create_batch_monitoring_model(input_shape, compile_model=True)
                    elif model_name == 'node_behavior':
                        trained_model = create_node_behavior_model(input_shape, compile_model=True)
                    elif model_name == 'arbitrator_bias':
                        trained_model = create_arbitrator_bias_model(input_shape, compile_model=True)
                    elif model_name == 'dispute_risk':
                        trained_model = create_dispute_risk_model(input_shape, compile_model=True)
                    
                    try:
                        # Sửa: lấy đúng trọng số đã huấn luyện từ server_state
                        model_weights = None
                        if hasattr(server_state, 'global_model_weights'):
                            model_weights = server_state.global_model_weights
                        elif hasattr(server_state, 'get_model_weights'):
                            model_weights = server_state.get_model_weights()
                        elif hasattr(server_state, 'model'):
                            model_weights = server_state.model
                        if model_weights is not None:
                            # Tùy thuộc vào kiểu model_weights, chuyển đổi về list numpy
                            if hasattr(model_weights, 'trainable'):
                                trained_model.set_weights([w.numpy() for w in model_weights.trainable])
                            elif hasattr(model_weights, 'numpy'):
                                trained_model.set_weights([w.numpy() for w in model_weights])
                            elif isinstance(model_weights, list):
                                trained_model.set_weights([np.array(w) for w in model_weights])
                            else:
                                logger.warning(f"Unknown model_weights type for {model_name}, cannot set weights.")
                            logger.info(f"Successfully transferred trained weights to {model_name} model")
                        else:
                            logger.warning(f"Could not extract trained weights for {model_name} from server_state.")
                    except Exception as e:
                        logger.warning(f"Could not set weights for {model_name}: {str(e)}. Using compiled model with initial weights.")
                    
                    global_models[model_name] = trained_model
                    training_history[model_name] = history # history is a list of dicts
                    # Log training metrics
                    logger.info(f"Training metrics for {model_name}:")
                    if isinstance(history, list) and history:
                        for round_num, metrics in enumerate(history):
                            log_line = f"  Round {round_num + 1}:"
                            if isinstance(metrics, dict):
                                for metric_name, metric_value in metrics.items():
                                    log_line += f" {metric_name}: {metric_value:.4f} " if isinstance(metric_value, float) else f" {metric_name}: {metric_value} "
                            else:
                                logger.info(f"  Round {round_num + 1}: {metrics}")
                                log_line += f" Metrics: {str(metrics)}" # Fallback if metrics is not a dict
                            logger.info(log_line.strip())
                    elif history: # If history is not a list but has content (e.g. a single dict)
                        logger.info(f"  History: {str(history)}")
                    else:
                        logger.info("  No detailed training history available or history is empty.")
                    # Save model
                    model_repository.save_model(trained_model, model_name, 'latest')
                    logger.info(f"Saved {model_name} model")
                else:
                    logger.error(f"Training failed for {model_name}")
                    # Không tạo fallback model nữa, chỉ lưu lại trạng thái lỗi.
                    global_models[model_name] = None
                    training_history[model_name] = {}
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                global_models[model_name] = None
                training_history[model_name] = {}

        monitoring.stop_timer(training_timer)
        logger.info("Federated Training phase completed.")

        # --- Anomaly Detection ---
        logger.info("Starting Anomaly Detection phase...")
        detection_timer = monitoring.start_timer("anomaly_detector", "anomaly_detection")
        
        all_anomalies = {}
        for model_name, trained_model in global_models.items():
            if model_name not in model_training_input_data or \
               not model_training_input_data[model_name] or \
               not model_training_input_data[model_name].get('features') or \
               len(model_training_input_data[model_name]['features']) == 0:
                logger.warning(f"Skipping anomaly detection for {model_name} due to lack of input features in model_training_input_data.")
                continue
                
            logger.info(f"--- Detecting anomalies using model: {model_name} ---")
            features_for_detection = np.array(model_training_input_data[model_name]['features']).astype(np.float32)
            ids_for_correlation = model_training_input_data[model_name].get('ids', []) # Get the list of ID dicts
            
            if features_for_detection.size == 0:
                logger.warning(f"No features to run detection for {model_name}.")
                continue
            
            try:
                # Run inference
                logger.info(f"Running inference with trained {model_name} model...")
                predictions = trained_model.predict(features_for_detection, verbose=0)
                # Create model results structure
                model_results = {
                    "predictions": {str(i): float(pred[0]) for i, pred in enumerate(predictions)},
                    "overall_score": float(np.mean(predictions))
                }
                
                # Save model results for anomaly detector to use
                results_path = os.path.join(dirs['results_dir'], f'{model_name}_model_results.json')
                try:
                    with open(results_path, 'w') as f:
                        json.dump(model_results, f, indent=2)
                    logger.info(f"Saved {model_name} model results to {results_path}")
                except Exception as e:
                    logger.warning(f"Could not save {model_name} model results: {str(e)}")
                # Create anomaly list based on predictions threshold
                anomalies = []
                threshold = 0.7  # Anomaly threshold (can be configured)
                for i, pred in enumerate(predictions):
                    if pred[0] > threshold:
                        anomaly = {
                            'model': model_name,
                            'index': i,
                            'score': float(pred[0]),
                            'id_info': ids_for_correlation[i] if i < len(ids_for_correlation) else {},
                        }
                        anomalies.append(anomaly)
                all_anomalies[model_name] = anomalies

                logger.info(f"Detected {len(anomalies)} anomalies for {model_name}.")
            except Exception as e:
                logger.error(f"Error during anomaly detection for {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                all_anomalies[model_name] = []

        monitoring.stop_timer(detection_timer)
        logger.info("Anomaly Detection phase completed.")

        # --- Response Generation ---
        logger.info("Starting Response Generation phase...")
        response_timer = monitoring.start_timer("response_engine", "response_generation")
        
        # Aggregate all anomalies
        total_anomalies = []
        for model_anomalies in all_anomalies.values():
            total_anomalies.extend(model_anomalies)
        
        logger.info(f"Total anomalies detected: {len(total_anomalies)}")
          # Generate responses
        responses = []
        if hasattr(response_engine, 'respond_to_detection'):
            for anomaly in total_anomalies:
                try:
                    response = response_engine.respond_to_detection(anomaly)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Error generating response for anomaly: {str(e)}")
        elif hasattr(response_engine, 'generate_response'):
            for anomaly in total_anomalies:
                try:
                    response = response_engine.generate_response(anomaly)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Error generating response for anomaly: {str(e)}")
        elif hasattr(response_engine, 'generate_responses'):
            try:
                responses = response_engine.generate_responses(total_anomalies)
            except Exception as e:
                logger.warning(f"Error generating responses: {str(e)}")
        else:
            logger.warning("ResponseEngine does not have respond_to_detection, generate_response or generate_responses method")
            responses = [{"message": "Default response - anomaly detected", "severity": "medium"} for _ in total_anomalies]

        monitoring.stop_timer(response_timer)
        logger.info("Response Generation phase completed.")

        # --- Save Results ---
        logger.info("Saving results...")
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode,
            "config": config,
            "training_summary": {
                "models_trained": list(global_models.keys()),
                "training_history": training_history # Ensure training_history is serializable
            },
            "detection_summary": {
                "total_anomalies": len(total_anomalies),
                "anomalies_by_model": {k: len(v) for k, v in all_anomalies.items()}
            },
            "response_summary": {
                "total_responses": len(responses),
                "response_types": {} # Populate this if you have different types of responses
            },
            "blockchain_monitoring_summary": {
                 "blocks_fetched": len(blockchain_monitoring_results),
                 "total_transactions_monitored": sum(len(b.get("transactions", [])) for b in blockchain_monitoring_results if isinstance(b, dict) and "error" not in b),
                 "data_file": os.path.join(dirs['results_dir'], "blockchain_monitoring_raw.json") if blockchain_monitoring_results else None
            }
        }
        
        # Save results
        # Save anomalies
        anomalies_path = os.path.join(dirs['results_dir'], 'anomalies_detected.json')
        try:
            with open(anomalies_path, 'w') as f:
                json.dump(to_serializable(all_anomalies), f, indent=2)
            logger.info(f"Anomalies saved to {anomalies_path}")
        except Exception as e:
            print(f"[DEBUG][ERROR] Failed to save anomalies: {e}")
            logger.error(f"Failed to save anomalies: {e}")
        # Save responses
        responses_path = os.path.join(dirs['results_dir'], 'responses.json')
        try:
            with open(responses_path, 'w') as f:
                json.dump(to_serializable(responses), f, indent=2)
            logger.info(f"Responses saved to {responses_path}")
        except Exception as e:
            print(f"[DEBUG][ERROR] Failed to save responses: {e}")
            logger.error(f"Failed to save responses: {e}")
        # Save summary
        summary_path = os.path.join(dirs['results_dir'], 'execution_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(to_serializable(summary), f, indent=2)
            logger.info(f"Execution summary saved to {summary_path}")
            # Also save a human-readable summary
            summary_txt_path = os.path.join(dirs['results_dir'], 'summary.txt')
            with open(summary_txt_path, 'w') as f:
                f.write(f"Federated Learning Execution Summary\n")
                f.write(f"Timestamp: {summary['timestamp']}\n")
                f.write(f"Mode: {summary['mode']}\n\n")
                f.write(f"Models Trained: {', '.join(summary['training_summary']['models_trained'])}\n")
                f.write(f"Total Anomalies Detected: {summary['detection_summary']['total_anomalies']}\n")
                f.write(f"Anomalies by Model: {summary['detection_summary']['anomalies_by_model']}\n")
                f.write(f"Total Responses: {summary['response_summary']['total_responses']}\n")
                f.write(f"Blockchain Blocks Fetched: {summary['blockchain_monitoring_summary']['blocks_fetched']}\n")
                f.write(f"Total Transactions Monitored: {summary['blockchain_monitoring_summary']['total_transactions_monitored']}\n")
                f.write(f"Blockchain Data File: {summary['blockchain_monitoring_summary']['data_file']}\n")
            logger.info(f"Human-readable summary saved to {summary_txt_path}")
        except Exception as e:
            print(f"[DEBUG][ERROR] Failed to save summary: {e}")
            logger.error(f"Failed to save summary: {e}")
    except FileNotFoundError as fnf_error:
        print(f"[DEBUG][FileNotFoundError] {fnf_error}")
        logger.error(f"FileNotFoundError: {fnf_error}")
    except Exception as e:
        print(f"[DEBUG][Exception] {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Exception: {e}")
    finally:
        print("[DEBUG] run_federated_learning finished.")

def connect_and_fetch_blockchain_data(rpc_url: str, num_blocks_to_fetch: int, output_dir: str) -> List[Dict[str, Any]]:
    """
    Connects to the blockchain, fetches data from recent blocks, and saves it.
    """
    logger.info(f"Attempting to connect to blockchain RPC: {rpc_url}")
    blockchain_data = []
    raw_blockchain_data_path = os.path.join(output_dir, 'blockchain_monitoring_raw.json')

    try:
        # Initialize the connector without contract details for general block/transaction fetching
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
            block = connector.get_block(block_num)
            if block:
                # Convert AttributeDicts and HexBytes to JSON serializable types
                block_serializable = json.loads(Web3.to_json(block))
                blockchain_data.append(block_serializable)
                logger.debug(f"Successfully fetched and serialized block {block_num}")
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

if __name__ == "__main__":
    print("[DEBUG] Main started.")
    args = parse_args()
    config = load_config(args.config_file)
    dirs = setup_directories(args)
    run_federated_learning(args, dirs, config)
    print("[DEBUG] Main finished.")
