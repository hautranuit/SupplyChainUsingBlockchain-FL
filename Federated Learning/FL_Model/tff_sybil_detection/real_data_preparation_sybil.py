#!/usr/bin/env python3
"""
Real data preparation for Sybil detection model in the ChainFLIP system.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional, Tuple

def generate_sybil_features(is_sybil: bool, num_features: int = 10, random_seed: int = None) -> np.ndarray:
    """
    Generate synthetic features for sybil/non-sybil nodes.
    
    Args:
        is_sybil: Whether this is a sybil node
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
    
    # Make sybil nodes have distinctive patterns
    if is_sybil:
        # More repetitive behavior
        features[0] = np.random.normal(3.0, 0.3)  # High activity frequency
        features[1] = np.random.normal(0.2, 0.1)  # Low uniqueness score
        features[2] = np.random.normal(2.5, 0.2)  # High correlation with other nodes
        features[3] = np.random.normal(0.1, 0.1)  # Low randomness in transactions
    else:
        # More organic behavior
        features[0] = np.random.normal(0.8, 0.4)  # Moderate activity frequency
        features[1] = np.random.normal(0.7, 0.2)  # Higher uniqueness score
        features[2] = np.random.normal(0.5, 0.3)  # Lower correlation with other nodes
        features[3] = np.random.normal(0.7, 0.2)  # Higher randomness in transactions
    
    return features

def extract_features_from_real_data(
    node_address: str,
    blockchain_events: Dict[str, List[Dict[str, Any]]], 
    demo_context: Dict[str, Any],
    all_node_addresses_in_fl: List[str], # For context like network density
    num_features: int = 15 # Increased number of features
) -> np.ndarray:
    """
    Extract features from real blockchain event data and demo_context.
    Enhanced to include more sophisticated features for Sybil/Bribery detection.

    Args:
        node_address: The Ethereum address of the node (lowercase).
        blockchain_events: A dictionary where keys are event names and values are lists of event data.
                           Event arguments are expected to be in event['args'].
        demo_context: Data from demo_context.json.
        all_node_addresses_in_fl: List of all node addresses participating in FL.
        num_features: The number of features to generate.

    Returns:
        A NumPy array of features.
    """
    features = np.zeros(num_features, dtype=np.float32)
    node_address_lower = node_address.lower()

    # Event Counts
    event_counts = {name: len(events) for name, events in blockchain_events.items()}

    # Feature 0: Total number of events involving the node (general activity level)
    features[0] = sum(event_counts.values())

    # Feature 1: Number of products minted by node (if manufacturer)
    # Assuming ProductMinted event has a 'minter' or 'manufacturer' arg
    product_minted_events = blockchain_events.get("ProductMinted", [])
    features[1] = sum(1 for event in product_minted_events if event.get('args', {}).get('minter', '').lower() == node_address_lower or event.get('args', {}).get('manufacturer', '').lower() == node_address_lower)

    # Feature 2: Number of products transferred from this node
    product_transferred_events = blockchain_events.get("ProductTransferred", [])
    features[2] = sum(1 for event in product_transferred_events if event.get('args', {}).get('from', '').lower() == node_address_lower)

    # Feature 3: Number of products transferred to this node
    features[3] = sum(1 for event in product_transferred_events if event.get('args', {}).get('to', '').lower() == node_address_lower)
    
    # Feature 4: Ratio of outgoing to incoming transfers (avoid division by zero)
    if features[3] > 0:
        features[4] = features[2] / features[3]
    elif features[2] > 0: # Has outgoing but no incoming
        features[4] = features[2] # Assign a high value
    else:
        features[4] = 0

    # Feature 5: Number of distinct counterparties in ProductTransferred events
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

    # Feature 9: Is Buyer (less likely a Sybil target, but good for context)
    buyer_keys = ["buyer1Address", "buyer2Address", "buyer3Address"]
    features[9] = 1.0 if any(demo_context.get(key, "").lower() == node_address_lower for key in buyer_keys) else 0.0

    # Feature 10: Number of disputes initiated by this node
    dispute_initiated_events = blockchain_events.get("DisputeInitiated", [])
    # Assuming DisputeInitiated has an 'initiator' arg
    features[10] = sum(1 for event in dispute_initiated_events if event.get('args', {}).get('initiator', '').lower() == node_address_lower)

    # Feature 11: Number of times this node was involved in a dispute (e.g., as defendant)
    # This requires knowing the structure of DisputeInitiated args, e.g., 'defendant' or 'productId' to trace back
    # Placeholder: for now, let's assume 'productId' can be used to link. This is highly speculative.
    # A more robust way would be to have specific fields like 'party1', 'party2'.
    # For now, this feature might be weak.
    disputed_product_ids = set(event.get('args', {}).get('productId') for event in dispute_initiated_events)
    node_product_interactions = 0
    # Simplistic check: if node transferred or received a product that was later disputed.
    for event in product_transferred_events:
        args = event.get('args', {})
        if args.get('tokenId') in disputed_product_ids and \
           (args.get('from', '').lower() == node_address_lower or args.get('to', '').lower() == node_address_lower):
            node_product_interactions +=1
    features[11] = node_product_interactions # This is a proxy, not direct involvement count.

    # Feature 12: Node Verified (from NodeVerified event)
    node_verified_events = blockchain_events.get("NodeVerified", [])
    features[12] = 1.0 if any(event.get('args', {}).get('nodeAddress', '').lower() == node_address_lower for event in node_verified_events) else 0.0

    # Feature 13: Activity within productDetails (e.g., as seller, recipient)
    product_details_activity = 0
    if "productDetails" in demo_context and isinstance(demo_context["productDetails"], list):
        for product in demo_context["productDetails"]:
            if isinstance(product, dict):
                if product.get("recipient", "").lower() == node_address_lower or \
                   product.get("currentOwnerAddress", "").lower() == node_address_lower or \
                   product.get("sellerAddress", "").lower() == node_address_lower or \
                   product.get("pendingBuyerAddress", "").lower() == node_address_lower or \
                   product.get("currentOwner", "").lower() == node_address_lower:
                    product_details_activity += 1
    features[13] = product_details_activity

    # Feature 14: Number of batches proposed by this node
    batch_proposed_events = blockchain_events.get("BatchProposed", [])
    # Assuming BatchProposed has a 'proposer' arg
    features[14] = sum(1 for event in batch_proposed_events if event.get('args', {}).get('proposer', '').lower() == node_address_lower)
    
    # Ensure features array has the correct length, pad with 0 if fewer features generated
    if len(features) < num_features:
        features = np.pad(features, (0, num_features - len(features)), 'constant')
    # Truncate if more features were generated (should not happen with explicit indexing)
    elif len(features) > num_features:
        features = features[:num_features]
        
    return features

def make_federated_data_sybil_real(
    all_node_addresses: List[str],
    blockchain_connector, 
    demo_context: Dict[str, Any], 
    num_fl_clients: int = 3,
    sybil_attack_log: Optional[Dict[str, Any]] = None,
    samples_per_client: int = 1, # Changed to 1, as we generate one feature vector per node
    num_features: int = 15 # Match the new feature count
) -> Tuple[List[tf.data.Dataset], Dict[str, List[str]]]:
    """
    Create federated data for Sybil detection model using real data.
    Each client gets data corresponding to one node.
    """
    sybil_addresses = []
    bribed_addresses = # Though this is sybil detection, bribery can be linked
    
    if sybil_attack_log:
        if "sybilNodes" in sybil_attack_log:
            for sybil_node in sybil_attack_log["sybilNodes"]:
                if "address" in sybil_node:
                    sybil_addresses.append(sybil_node["address"].lower())
        
        if "scenarioD" in sybil_attack_log and "bribedNodes" in sybil_attack_log["scenarioD"]:
            for bribed_node in sybil_attack_log["scenarioD"]["bribedNodes"]:
                if "address" in bribed_node:
                    bribed_addresses.append(bribed_node["address"].lower())
    
    federated_data = []
    client_node_mapping = {} 

    all_node_addresses_lower = [addr.lower() for addr in all_node_addresses]
    
    # Each node address will be a client if num_fl_clients allows, or a subset.
    # For robust FL, typically you'd want more clients than this if you have many nodes.
    # Here, we map each selected node to be its own FL client.
    
    if not all_node_addresses_lower:
        print("Warning: No node addresses provided. Cannot generate federated data.")
        return [], {}

    # Determine the actual number of clients based on available nodes and num_fl_clients
    # If num_fl_clients is more than available nodes, use all nodes as clients.
    # If num_fl_clients is less, sample from the available nodes.
    if len(all_node_addresses_lower) <= num_fl_clients:
        selected_client_node_addresses = all_node_addresses_lower
    else:
        selected_client_node_addresses = random.sample(all_node_addresses_lower, num_fl_clients)

    if not selected_client_node_addresses:
        print("Warning: No client node addresses selected. Cannot generate federated data.")
        return [], {}

    for client_id, client_node_address in enumerate(selected_client_node_addresses):
        client_node_mapping[str(client_id)] = [client_node_address]

        # Fetch all relevant events for this node_address
        relevant_event_names = [
            "ProductRegistered", "ProductTransferred", "ProductStateChanged", 
            "PaymentMade", "NodeRegistered", "NodeVerified", "ProductMinted",
            "DisputeInitiated", "BatchProposed", "BatchValidated", "CIDStored",
            "DirectSaleAndTransferCompleted", "PaymentAndTransferCompleted"
            # Add other relevant events from your contract
        ]
        
        node_blockchain_events = {}
        for event_name in relevant_event_names:
            # Assuming SupplyChain contract is the primary one.
            # The get_events might need to be more sophisticated if events don't directly filter by a single node address field.
            # For now, we fetch all events of that type and filter within extract_features_from_real_data.
            # A more optimized approach would be to pass argument_filters to get_events if possible,
            # e.g., filters that check if client_node_address is in 'from', 'to', 'minter', 'proposer', etc.
            # This depends on the capabilities of get_events and the structure of event arguments.
            node_blockchain_events[event_name] = blockchain_connector.get_events(
                contract_name="SupplyChainNFT", # Ensure this is the correct contract alias used in BlockchainConnector
                event_name=event_name,
                from_block=0 
            )
            # Post-filter if get_events doesn't support complex argument filtering for all fields
            # This is less efficient but ensures we only pass relevant events to feature extraction
            # For example:
            # filtered_events = []
            # for evt in node_blockchain_events[event_name]:
            #     args = evt.get('args', {})
            #     if any(val.lower() == client_node_address for val in args.values() if isinstance(val, str)):
            #         filtered_events.append(evt)
            # node_blockchain_events[event_name] = filtered_events
            # The current extract_features_from_real_data does its own filtering based on node_address, so this explicit post-filter might be redundant here.


        features_for_node = extract_features_from_real_data(
            client_node_address, 
            node_blockchain_events, 
            demo_context,
            all_node_addresses_lower, # Pass all participating FL node addresses
            num_features
        )
        
        is_sybil = client_node_address in sybil_addresses
        is_bribed = client_node_address in bribed_addresses # Consider if bribed nodes should also be labeled as "1" for Sybil model
        label = 1.0 if is_sybil or is_bribed else 0.0 # Label as 1 if Sybil or Bribed
        
        # Each client has one data point (its own feature vector and label)
        # To create a tf.data.Dataset, we need at least one sample.
        features_array = np.array([features_for_node], dtype=np.float32) # Shape (1, num_features)
        labels_array = np.array([[label]], dtype=np.float32) # Shape (1, 1)
        
        dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
        # Batching and repeating:
        # - Batch size of 1 since each client has only one primary data point.
        # - Repeat is important for TFF training loops that expect multiple batches.
        dataset = dataset.batch(1).repeat(100) # Repeat significantly for multiple rounds of training
        
        federated_data.append(dataset)
    
    if not federated_data:
        print("Warning: No federated data generated. Returning empty list and mapping.")
        return [], {}

    return federated_data, client_node_mapping
