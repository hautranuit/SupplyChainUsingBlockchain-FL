#!/usr/bin/env python3
"""
Feature extractor for blockchain data in the ChainFLIP Federated Learning system.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extractor that processes blockchain data and demo context
    to generate features for different model types in the federated learning system.
    """
    
    def __init__(self, num_features: int = 15):
        """
        Initialize the feature extractor.
        
        Args:
            num_features: Number of features to extract per sample
        """
        self.num_features = num_features
    
    def extract_features(self, fl_input_data: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
        """
        Extract features from blockchain data for different model types.
        
        Args:
            fl_input_data: Dictionary containing blockchain events and demo context
            
        Returns:
            Dictionary with extracted features and labels for each model type
        """
        logger.info("Starting feature extraction from blockchain data...")
        
        blockchain_events = fl_input_data.get('blockchain_events', {})
        demo_context = fl_input_data.get('demo_context', {})
        all_node_addresses = fl_input_data.get('node_addresses', [])
        
        # Extract features for different model types
        processed_data = {
            'sybil_detection': self._extract_sybil_features(
                blockchain_events, demo_context, all_node_addresses
            ),
            'batch_monitoring': self._extract_batch_features(
                blockchain_events, demo_context, all_node_addresses
            ),
            'node_behavior': self._extract_node_behavior_features(
                blockchain_events, demo_context, all_node_addresses
            ),
            'dispute_risk': self._extract_dispute_features(
                blockchain_events, demo_context, all_node_addresses
            ),
            'arbitrator_bias': self._extract_arbitrator_features(
                blockchain_events, demo_context, all_node_addresses
            )
        }
        
        logger.info(f"Feature extraction completed for {len(processed_data)} model types")
        return processed_data
    
    def _extract_sybil_features(self, blockchain_events: Dict, demo_context: Dict, 
                               all_node_addresses: List[str]) -> Dict[str, List]:
        """Extract features for Sybil detection model."""
        logger.info("Extracting Sybil detection features...")
        
        features = []
        labels = []
        
        for node_address in all_node_addresses:
            node_features = self._extract_features_from_real_data(
                node_address, blockchain_events, demo_context, all_node_addresses
            )
            features.append(node_features.tolist())
            
            # For normal scenario, all nodes are labeled as non-sybil (0)
            # In attack scenarios, this would be determined by attack logs
            labels.append(0)
        
        return {'features': features, 'labels': labels}
    
    def _extract_batch_features(self, blockchain_events: Dict, demo_context: Dict,
                               all_node_addresses: List[str]) -> Dict[str, List]:
        """Extract features for batch monitoring model."""
        logger.info("Extracting batch monitoring features...")
        
        features = []
        labels = []
        
        # Generate synthetic batch features based on available data
        num_samples = max(50, len(all_node_addresses) * 2)
        
        for i in range(num_samples):
            batch_features = self._generate_batch_features(
                is_anomalous=False,  # Normal scenario
                num_features=10
            )
            features.append(batch_features.tolist())
            labels.append(0)  # Normal batch
        
        return {'features': features, 'labels': labels}
    
    def _extract_node_behavior_features(self, blockchain_events: Dict, demo_context: Dict,
                                       all_node_addresses: List[str]) -> Dict[str, List]:
        """Extract features for node behavior analysis."""
        logger.info("Extracting node behavior features...")
        
        features = []
        labels = []
        
        for node_address in all_node_addresses:
            # Generate time-series features for node behavior
            behavior_features = self._generate_timeseries_features(
                is_anomalous=False,  # Normal scenario
                num_features=10
            )
            features.append(behavior_features.tolist())
            labels.append(0)  # Normal behavior
        
        return {'features': features, 'labels': labels}
    
    def _extract_dispute_features(self, blockchain_events: Dict, demo_context: Dict,
                                 all_node_addresses: List[str]) -> Dict[str, List]:
        """Extract features for dispute risk assessment."""
        logger.info("Extracting dispute risk features...")
        
        features = []
        labels = []
        
        # Extract features based on dispute-related events
        num_samples = max(30, len(all_node_addresses))
        
        for i in range(num_samples):
            dispute_features = self._generate_dispute_features(
                is_dispute_risk=False,  # Normal scenario
                num_features=6
            )
            features.append(dispute_features.tolist())
            labels.append(0)  # Low dispute risk
        
        return {'features': features, 'labels': labels}
    
    def _extract_arbitrator_features(self, blockchain_events: Dict, demo_context: Dict,
                                    all_node_addresses: List[str]) -> Dict[str, List]:
        """Extract features for arbitrator bias detection."""
        logger.info("Extracting arbitrator bias features...")
        
        features = []
        labels = []
        
        # Extract arbitrator-related features
        arbitrator_events = blockchain_events.get("ArbitratorSelected", [])
        
        if arbitrator_events:
            for event in arbitrator_events[:20]:  # Limit samples
                arb_features = np.random.normal(0.5, 0.2, 5).astype(np.float32)
                features.append(arb_features.tolist())
                labels.append(0)  # No bias in normal scenario
        else:
            # Generate minimal sample data if no arbitrator events
            for i in range(10):
                arb_features = np.random.normal(0.5, 0.2, 5).astype(np.float32)
                features.append(arb_features.tolist())
                labels.append(0)
        
        return {'features': features, 'labels': labels}
    
    def _extract_features_from_real_data(self, node_address: str, 
                                        blockchain_events: Dict[str, List[Dict[str, Any]]], 
                                        demo_context: Dict[str, Any],
                                        all_node_addresses_in_fl: List[str]) -> np.ndarray:
        """
        Extract features from real blockchain event data and demo_context.
        Enhanced to include sophisticated features for Sybil/Bribery detection.
        """
        features = np.zeros(self.num_features, dtype=np.float32)
        node_address_lower = node_address.lower()

        # Event Counts
        event_counts = {name: len(events) for name, events in blockchain_events.items()}

        # Feature 0: Total number of events involving the node
        features[0] = sum(event_counts.values())

        # Feature 1: Number of products minted by node
        product_minted_events = blockchain_events.get("ProductMinted", [])
        features[1] = sum(1 for event in product_minted_events 
                         if event.get('args', {}).get('minter', '').lower() == node_address_lower or 
                            event.get('args', {}).get('manufacturer', '').lower() == node_address_lower)

        # Feature 2: Number of products transferred from this node
        product_transferred_events = blockchain_events.get("ProductTransferred", [])
        features[2] = sum(1 for event in product_transferred_events 
                         if event.get('args', {}).get('from', '').lower() == node_address_lower)

        # Feature 3: Number of products transferred to this node
        features[3] = sum(1 for event in product_transferred_events 
                         if event.get('args', {}).get('to', '').lower() == node_address_lower)
        
        # Feature 4: Ratio of outgoing to incoming transfers
        if features[3] > 0:
            features[4] = features[2] / features[3]
        elif features[2] > 0:
            features[4] = features[2]
        else:
            features[4] = 0

        # Feature 5: Number of distinct counterparties
        counterparties = set()
        for event in product_transferred_events:
            args = event.get('args', {})
            if args.get('from', '').lower() == node_address_lower and args.get('to', ''):
                counterparties.add(args['to'].lower())
            elif args.get('to', '').lower() == node_address_lower and args.get('from', ''):
                counterparties.add(args['from'].lower())
        features[5] = len(counterparties)

        # Role-based features from demo_context
        # Feature 6: Is Manufacturer
        features[6] = 1.0 if demo_context.get("manufacturerAddress", "").lower() == node_address_lower else 0.0
        
        # Feature 7: Is Transporter
        transporter_keys = ["transporter1Address", "transporter2Address", "transporter3Address"]
        features[7] = 1.0 if any(demo_context.get(key, "").lower() == node_address_lower for key in transporter_keys) else 0.0

        # Feature 8: Is Retailer
        retailer_keys = ["retailerAddress", "retailer1Address", "retailer2Address", "retailer3Address"]
        features[8] = 1.0 if any(demo_context.get(key, "").lower() == node_address_lower for key in retailer_keys) else 0.0

        # Feature 9: Is Buyer
        buyer_keys = ["buyer1Address", "buyer2Address", "buyer3Address"]
        features[9] = 1.0 if any(demo_context.get(key, "").lower() == node_address_lower for key in buyer_keys) else 0.0

        # Feature 10: Number of disputes initiated by this node
        dispute_initiated_events = blockchain_events.get("DisputeInitiated", [])
        features[10] = sum(1 for event in dispute_initiated_events 
                          if event.get('args', {}).get('initiator', '').lower() == node_address_lower)

        # Feature 11: Node involvement in disputed products (proxy)
        disputed_product_ids = set(event.get('args', {}).get('productId') for event in dispute_initiated_events)
        node_product_interactions = 0
        for event in product_transferred_events:
            args = event.get('args', {})
            if (args.get('tokenId') in disputed_product_ids and 
                (args.get('from', '').lower() == node_address_lower or args.get('to', '').lower() == node_address_lower)):
                node_product_interactions += 1
        features[11] = node_product_interactions

        # Feature 12: Node Verified status
        node_verified_events = blockchain_events.get("NodeVerified", [])
        features[12] = 1.0 if any(event.get('args', {}).get('nodeAddress', '').lower() == node_address_lower 
                                 for event in node_verified_events) else 0.0

        # Feature 13: Activity within productDetails
        product_details_activity = 0
        if "productDetails" in demo_context and isinstance(demo_context["productDetails"], list):
            for product in demo_context["productDetails"]:
                if isinstance(product, dict):
                    if (product.get("recipient", "").lower() == node_address_lower or
                        product.get("currentOwnerAddress", "").lower() == node_address_lower or
                        product.get("sellerAddress", "").lower() == node_address_lower or
                        product.get("pendingBuyerAddress", "").lower() == node_address_lower or
                        product.get("currentOwner", "").lower() == node_address_lower):
                        product_details_activity += 1
        features[13] = product_details_activity

        # Feature 14: Number of batches proposed by this node
        batch_proposed_events = blockchain_events.get("BatchProposed", [])
        features[14] = sum(1 for event in batch_proposed_events 
                          if event.get('args', {}).get('proposer', '').lower() == node_address_lower)
        
        # Ensure features array has the correct length
        if len(features) < self.num_features:
            features = np.pad(features, (0, self.num_features - len(features)), 'constant')
        elif len(features) > self.num_features:
            features = features[:self.num_features]
            
        return features
    
    def _generate_batch_features(self, is_anomalous: bool, num_features: int = 10) -> np.ndarray:
        """Generate synthetic features for batch monitoring."""
        features = np.random.normal(size=num_features).astype(np.float32)
        
        if is_anomalous:
            # Features indicating anomalies
            features[0] = np.random.normal(1.8, 0.3)  # High temperature variation
            features[1] = np.random.normal(1.5, 0.4)  # High humidity variation
            features[2] = np.random.normal(0.2, 0.1)  # Low quality score
            features[3] = np.random.normal(1.7, 0.3)  # High time deviation
        else:
            # Normal batch features
            features[0] = np.random.normal(0.5, 0.2)  # Normal temperature variation
            features[1] = np.random.normal(0.6, 0.2)  # Normal humidity variation
            features[2] = np.random.normal(0.8, 0.1)  # Normal quality score
            features[3] = np.random.normal(0.4, 0.2)  # Normal time deviation
        
        return features
    
    def _generate_timeseries_features(self, is_anomalous: bool, num_features: int = 10) -> np.ndarray:
        """Generate synthetic timeseries features for node behavior analysis."""
        features = np.random.normal(size=num_features).astype(np.float32)
        
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
    
    def _generate_dispute_features(self, is_dispute_risk: bool, num_features: int = 10) -> np.ndarray:
        """Generate synthetic features for dispute risk assessment."""
        features = np.random.normal(size=num_features).astype(np.float32)
        
        if is_dispute_risk:
            # Features indicating high dispute risk
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
