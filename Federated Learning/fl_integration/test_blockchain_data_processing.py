#!/usr/bin/env python3
"""
Test script for blockchain data processing with FL integration.
Tests the updated data processor with real blockchain interaction data.
"""

import json
import sys
import os
import logging
from pathlib import Path

# Add the data_processor directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_processor'))

from processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_blockchain_processor")

def load_demo_context(demo_context_path: str):
    """Load the demo_context.json file"""
    try:
        with open(demo_context_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded demo context from {demo_context_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading demo context: {e}")
        return None

def analyze_blockchain_interactions(data):
    """Analyze the types and patterns of blockchain interactions"""
    if not data or 'nodes' not in data:
        logger.error("Invalid data structure")
        return
    
    logger.info("=== Blockchain Interaction Analysis ===")
    
    all_interaction_types = set()
    total_interactions = 0
    nodes_with_interactions = 0
    
    for node_addr, node_info in data['nodes'].items():
        interactions = node_info.get('interactions', [])
        if interactions:
            nodes_with_interactions += 1
            total_interactions += len(interactions)
            
            for interaction in interactions:
                interaction_type = interaction.get('type', 'Unknown')
                all_interaction_types.add(interaction_type)
    
    logger.info(f"Total nodes: {len(data['nodes'])}")
    logger.info(f"Nodes with interactions: {nodes_with_interactions}")
    logger.info(f"Total interactions: {total_interactions}")
    logger.info(f"Unique interaction types: {len(all_interaction_types)}")
    logger.info(f"Interaction types found: {sorted(all_interaction_types)}")
    
    # Analyze interaction frequency by type
    type_counts = {}
    for node_addr, node_info in data['nodes'].items():
        for interaction in node_info.get('interactions', []):
            interaction_type = interaction.get('type', 'Unknown')
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
    
    logger.info("\nInteraction frequency by type:")
    for interaction_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {interaction_type}: {count}")

def test_data_processor(data):
    """Test the data processor with blockchain data"""
    logger.info("=== Testing Data Processor ===")
    
    processor = DataProcessor(cache_dir="./test_cache")
    
    try:
        # Process the data
        processed_data = processor.preprocess_data(data)
        
        logger.info("Data processing completed successfully!")
        
        # Analyze results
        for model_name, model_data in processed_data.items():
            features = model_data['features']
            labels = model_data['labels']
            
            logger.info(f"\n{model_name.upper()} Model:")
            logger.info(f"  Samples: {len(features)}")
            logger.info(f"  Features per sample: {len(features[0]) if features else 0}")
            logger.info(f"  Positive labels: {sum(labels)} / {len(labels)}")
            logger.info(f"  Label distribution: {sum(labels)/len(labels)*100:.1f}% positive" if labels else "No labels")
            
            if features:
                logger.info(f"  Sample feature vector: {features[0]}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_synthetic_data_generation():
    """Test synthetic data generation"""
    logger.info("=== Testing Synthetic Data Generation ===")
    
    processor = DataProcessor(cache_dir="./test_cache")
    
    try:
        # Generate synthetic context
        synthetic_context = processor.create_synthetic_blockchain_context(base_samples=10)
        
        logger.info(f"Generated synthetic context with {len(synthetic_context['nodes'])} nodes")
        
        # Process synthetic data
        processed_synthetic = processor.preprocess_data(synthetic_context)
        
        for model_name, model_data in processed_synthetic.items():
            features = model_data['features']
            labels = model_data['labels']
            logger.info(f"Synthetic {model_name}: {len(features)} samples")
        
        return synthetic_context
        
    except Exception as e:
        logger.error(f"Error in synthetic data generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_processed_data(processed_data, output_path):
    """Save processed data to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, cls=processor.NumpyJSONEncoder)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")

def main():
    logger.info("Starting blockchain data processing test...")
    
    # Path to demo context
    demo_context_path = "../../SupplyChain_dapp/scripts/lifecycle_demo/demo_context.json"
    
    # Check if demo context exists
    if not os.path.exists(demo_context_path):
        logger.warning(f"Demo context not found at {demo_context_path}")
        logger.info("Testing with synthetic data only...")
        
        # Test synthetic data generation
        synthetic_context = test_synthetic_data_generation()
        if synthetic_context:
            logger.info("Synthetic data generation test completed successfully!")
        return
    
    # Load real data
    data = load_demo_context(demo_context_path)
    if not data:
        logger.error("Failed to load demo context")
        return
    
    # Analyze the data
    analyze_blockchain_interactions(data)
    
    # Test data processing
    processed_data = test_data_processor(data)
    
    if processed_data:
        # Save results
        output_dir = "./test_results"
        os.makedirs(output_dir, exist_ok=True)
        
        save_processed_data(processed_data, f"{output_dir}/processed_blockchain_data.json")
        
        logger.info("=== Test Summary ===")
        total_samples = sum(len(model_data['features']) for model_data in processed_data.values())
        logger.info(f"Total processed samples across all models: {total_samples}")
        
        if total_samples >= 90:  # 30 samples per model minimum
            logger.info("✅ SUCCESS: All models have sufficient samples for FL training!")
        else:
            logger.warning("⚠️  WARNING: Some models may not have enough samples")
    
    # Test synthetic data generation
    logger.info("\n" + "="*50)
    test_synthetic_data_generation()

if __name__ == "__main__":
    main()
