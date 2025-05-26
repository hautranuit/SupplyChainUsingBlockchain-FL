import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from model_repository.repository import ModelRepository
from preprocessing.feature_extractor import FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference for all FL models')
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json'))
    parser.add_argument('--input-data-file', type=str, default=default_input, required=False, help='Path to input data file (e.g., demo_context.json)')
    parser.add_argument('--models-dir', type=str, default=os.path.join(os.path.dirname(__file__), 'models'), help='Directory containing trained models')
    parser.add_argument('--output-file', type=str, default='inference_results.json', help='File to save inference results')
    parser.add_argument('--model-version', type=str, default=None, help='Model version to load (default: latest)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)
    logger = logging.getLogger("run_inference")

    # Load input data
    with open(args.input_data_file, 'r') as f:
        input_data = json.load(f)
    logger.info(f"Loaded input data from {args.input_data_file}")

    # Extract features for all models
    feature_extractor = FeatureExtractor()
    processed_data = feature_extractor.extract_features(input_data)    # List of core attack detection models to run inference
    model_names = [
        'sybil_detection',
        'bribery_detection'
    ]

    # Load models
    model_repo = ModelRepository(args.models_dir)
    results = {}
    for model_name in model_names:
        logger.info(f"Loading model: {model_name}")
        model = model_repo.load_model(model_name, version=args.model_version)
        if model is None:
            logger.warning(f"Model {model_name} not found. Skipping.")
            continue
        # Prepare features
        df = processed_data.get(model_name)
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(f"No data for model {model_name}. Skipping.")
            continue
        # Extract features (exclude IDs and label)
        id_cols = [col for col in df.columns if col.endswith('_id') or col in ['node_id','batch_id','validator_id','arbitrator_id','dispute_id']]
        feature_cols = [col for col in df.columns if col not in id_cols and col != 'label']
        features = df[feature_cols].values.astype(np.float32)
        # Run inference
        preds = model.predict(features, verbose=0)
        # Save results
        results[model_name] = {
            'predictions': preds.flatten().tolist(),
            'ids': df[id_cols].to_dict(orient='records') if id_cols else [],
            'threshold': 0.7,  # Default threshold, adjust as needed
            'anomalies': [i for i, p in enumerate(preds.flatten()) if p >= 0.7]
        }
        logger.info(f"Model {model_name}: {len(results[model_name]['anomalies'])} anomalies detected.")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Inference results saved to {args.output_file}")

if __name__ == "__main__":
    main()
