import os
import numpy as np
import logging
from data_processor.processor import DataProcessor

logger = logging.getLogger("feature_extractor")

class FeatureExtractor:
    def __init__(self, cache_dir: str = "./cache"):
        self.data_processor = DataProcessor(cache_dir=cache_dir)

    def extract_features(self, input_data):
        """
        Extract features for all FL models from input_data (e.g., demo_context.json structure).
        Returns a dict with keys for each model, each containing a DataFrame (features + label + id columns if available).
        """
        import pandas as pd
        try:
            processed = self.data_processor.preprocess_data(input_data)
            logger.info(f"Feature extraction complete. Models: {list(processed.keys())}")
            processed_df = {}
            for model, data in processed.items():
                # If data is dict with 'features' and 'labels', convert to DataFrame
                if isinstance(data, dict) and 'features' in data and 'labels' in data:
                    features = data['features']
                    labels = data['labels']
                    # If 'ids' (list of dict) exists, add to DataFrame
                    ids = data.get('ids', None)
                    if features and labels and len(features) == len(labels):
                        df = pd.DataFrame(features)
                        df['label'] = labels
                        if ids and isinstance(ids, list) and len(ids) == len(features):
                            ids_df = pd.DataFrame(ids)
                            ids_df = ids_df.reset_index(drop=True)
                            df = pd.concat([df.reset_index(drop=True), ids_df], axis=1)
                        processed_df[model] = df
                    else:
                        # Return empty DataFrame with label column
                        processed_df[model] = pd.DataFrame(columns=[f'f{i}' for i in range(1, 16)] + ['label'])
                elif isinstance(data, pd.DataFrame):
                    processed_df[model] = data
                else:
                    # Return empty DataFrame if format is incorrect
                    processed_df[model] = pd.DataFrame()
            return processed_df
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
