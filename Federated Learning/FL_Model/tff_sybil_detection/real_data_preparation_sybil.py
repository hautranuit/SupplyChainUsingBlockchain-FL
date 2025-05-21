# Real Data Preparation for Sybil Detection FL Model

import tensorflow as tf
import numpy as np
import time
import sys
import os
from dotenv import load_dotenv
import json

# Adjust path to import connectors
current_dir = os.path.dirname(os.path.abspath(__file__))
fl_model_dir = os.path.dirname(current_dir)
sys.path.append(fl_model_dir)

from connectors.blockchain_connector import BlockchainConnector

# Define feature specification for TFF, consistent with the Sybil detection model_definition.py
NUM_SYBIL_FEATURES = 5
ELEMENT_SPEC_SYBIL = (
    tf.TensorSpec(shape=(NUM_SYBIL_FEATURES,), dtype=tf.float32),
    tf.TensorSpec(shape=(1,), dtype=tf.int32)     # Labels (0 for normal, 1 for Sybil)
)

RPC_URL_OVERRIDE = "https://rpc-amoy.polygon.technology/"

# Heuristic thresholds for Sybil labeling (can be tuned)
SYBIL_MAX_AGE_SECONDS = 86400 * 7  # 7 ngày
SYBIL_MAX_REPUTATION = 30  # Giảm ngưỡng reputation
SYBIL_MAX_TX_COUNT = 5  # Tăng ngưỡng số giao dịch
SYBIL_MAX_COUNTERPARTIES = 2  # Tăng ngưỡng số đối tác

def get_node_registration_timestamp(bc_connector, node_address):
    try:
        checksum_node_address = bc_connector.w3.to_checksum_address(node_address)
        node_verified_events = bc_connector.get_events(
            contract_name="SupplyChainNFT",
            event_name="NodeVerified",
            argument_filters={"node": checksum_node_address},
            from_block=0
        )
        if node_verified_events:
            node_verified_events.sort(key=lambda x: x["blockNumber"])
            first_verification_event = node_verified_events[0]
            block_number = first_verification_event["blockNumber"]
            timestamp = bc_connector.w3.eth.get_block(block_number)["timestamp"]
            return timestamp
        return 0
    except Exception as e:
        print(f"Error getting registration timestamp for {node_address}: {e}")
        return 0

def get_node_transaction_count_and_counterparties(bc_connector, node_address):
    transaction_count = 0
    counterparties = set()
    try:
        checksum_node_address = bc_connector.w3.to_checksum_address(node_address)
        event_types_and_roles = [
            ("ProductMinted", "owner"),
            ("InitialCIDStored", "actor"),
            ("DirectSaleAndTransferCompleted", "seller", "buyer"),
            ("PaymentAndTransferCompleted", "seller", "buyer"),
            ("DisputeInitiated", "initiator", "currentOwner") # currentOwner is the other party
        ]
        for event_info in event_types_and_roles:
            event_name = event_info[0]
            roles = event_info[1:]
            for role in roles:
                # Fetch events where the node is in this role
                filters = {role: checksum_node_address}
                events = bc_connector.get_events(
                    contract_name="SupplyChainNFT",
                    event_name=event_name,
                    argument_filters=filters,
                    from_block=0
                )
                transaction_count += len(events)
                # Identify counterparties
                for event_data in events:
                    for other_role in roles:
                        if other_role != role and other_role in event_data["args"]:
                            counterparty = event_data["args"][other_role]
                            if isinstance(counterparty, str) and bc_connector.w3.to_checksum_address(counterparty) != checksum_node_address:
                                counterparties.add(bc_connector.w3.to_checksum_address(counterparty))
        if checksum_node_address in counterparties:
            counterparties.remove(checksum_node_address)
        return transaction_count, len(counterparties)
    except Exception as e:
        print(f"Error in get_node_transaction_count_and_counterparties for {node_address}: {e}")
        return 0, 0

def generate_features_and_label_for_sybil(bc_connector, node_address, sybil_attack_log=None, fl_model=None):
    features = np.zeros(NUM_SYBIL_FEATURES, dtype=np.float32)
    current_timestamp = int(time.time())
    is_known_sybil = False
    sybil_node_details = None

    # Check if this node is a known Sybil from the log
    if sybil_attack_log and "sybilNodes" in sybil_attack_log:
        for node_info in sybil_attack_log["sybilNodes"]:
            if node_info.get("address") == node_address:
                is_known_sybil = True
                sybil_node_details = node_info
                print(f"  Node {node_address} identified as a known Sybil from sybil_attack_log.json.")
                break

    # Feature 1: Age (can use initialization timestamp from log if available for Sybils)
    reg_timestamp = 0
    if is_known_sybil and sybil_node_details and "initialization" in sybil_node_details and sybil_node_details["initialization"]:
        # Assuming the first initialization action has a timestamp
        # Find the earliest timestamp from initialization actions if multiple exist
        init_timestamps = [action.get("timestamp") for action in sybil_node_details["initialization"] if action.get("timestamp")]
        if init_timestamps:
            # Timestamps in the log are ISO format strings, e.g., "2024-07-30T10:00:00.123Z"
            # Convert to Unix timestamp (seconds since epoch)
            try:
                # Take the first one as the primary registration/initialization time for this Sybil node
                iso_ts_str = init_timestamps[0]
                # Handle potential 'Z' for UTC
                if iso_ts_str.endswith('Z'):
                    iso_ts_str = iso_ts_str[:-1] + '+00:00'
                reg_timestamp = int(datetime.fromisoformat(iso_ts_str).timestamp())
                print(f"  Using initialization timestamp from log for Sybil {node_address}: {reg_timestamp}")
            except ValueError as ve:
                print(f"  Error parsing timestamp from log for {node_address}: {ve}. Falling back to blockchain query.")
                reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)
        else:
            reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)
    else:
        reg_timestamp = get_node_registration_timestamp(bc_connector, node_address)

    age_seconds = 0
    if reg_timestamp > 0 and reg_timestamp < current_timestamp:
        age_seconds = float(current_timestamp - reg_timestamp)
        features[0] = age_seconds
    else:
        features[0] = 0.0

    # Feature 2: Reputation (can use initial reputation from log if available for Sybils)
    reputation = 0
    if is_known_sybil and sybil_node_details and "initialization" in sybil_node_details:
        # Look for adminUpdateReputation in initialization actions
        rep_action = next((act for act in sybil_node_details["initialization"] if act.get("action") == "adminUpdateReputation"), None)
        if rep_action and "newReputation" in rep_action:
            reputation = float(rep_action["newReputation"])
            print(f"  Using initial reputation from log for Sybil {node_address}: {reputation}")
        else:
            reputation_bc = bc_connector.get_node_reputation(node_address)
            reputation = float(reputation_bc if reputation_bc is not None else 0)
    else:
        reputation_bc = bc_connector.get_node_reputation(node_address)
        reputation = float(reputation_bc if reputation_bc is not None else 0)
    features[1] = reputation
    
    # Feature 3 & 4: Transaction count and distinct counterparties
    # For Sybils, we could potentially use their logged activities for a more precise count if desired,
    # but blockchain query reflects their overall on-chain footprint which is also relevant.
    # For now, stick to blockchain query for these for all nodes for consistency.
    tx_count, distinct_counterparties = get_node_transaction_count_and_counterparties(bc_connector, node_address)
    features[2] = float(tx_count)
    features[3] = float(distinct_counterparties)

    # Feature 5: Verification Status (Sybils might be set as verified during simulation)
    is_verified_status = bc_connector.is_node_verified(node_address)
    features[4] = 1.0 if is_verified_status else 0.0

    # Labeling: If it's a known Sybil from the log, label it as 1 (Sybil).
    # Otherwise, use heuristic rules or FL model prediction.
    if is_known_sybil:
        label_val = 1
        print(f"  Node {node_address} labeled as SYBIL (1) based on sybil_attack_log.json.")
    elif fl_model is not None:
        try:
            sybil_probability = fl_model.predict(features) # Assuming fl_model.predict expects a single feature set
            label_val = 1 if sybil_probability > 0.5 else 0
            print(f"  Node {node_address} - FL Model Prediction: {sybil_probability:.4f}, Label: {'SYBIL' if label_val == 1 else 'NORMAL'}")
        except Exception as e:
            print(f"Error using FL model for {node_address}: {e}. Falling back to heuristic rules.")
            label_val = apply_heuristic_rules(features, is_verified_status, reputation)
    else:
        label_val = apply_heuristic_rules(features, is_verified_status, reputation)
        print(f"  Node {node_address} labeled as {'SYBIL' if label_val == 1 else 'NORMAL'} ({label_val}) based on heuristic rules.")
    
    label = np.array([label_val], dtype=np.int32)
    return features, label

def apply_heuristic_rules(features, is_verified_status, reputation):
    """Apply heuristic rules to determine if a node is Sybil"""
    sybil_score = 0
    
    # Heuristic rules với trọng số
    if not is_verified_status:
        sybil_score += 2  # Tăng trọng số cho việc chưa verify
    
    if features[0] < SYBIL_MAX_AGE_SECONDS and features[0] > 0:
        sybil_score += 1
    elif features[0] == 0 and is_verified_status:
        sybil_score += 0

    if (reputation if reputation is not None else 0) < SYBIL_MAX_REPUTATION:
        sybil_score += 1.5  # Tăng trọng số cho reputation thấp
    
    if features[2] < SYBIL_MAX_TX_COUNT:
        sybil_score += 1
    
    if features[3] < SYBIL_MAX_COUNTERPARTIES:
        sybil_score += 1

    # Điều chỉnh ngưỡng để phù hợp hơn
    if is_verified_status and (reputation if reputation is not None else 0) > SYBIL_MAX_REPUTATION:
        return 0  # Node đã verify và có reputation cao
    else:
        return 1 if sybil_score >= 4 else 0  # Tăng ngưỡng để giảm false positive

def load_real_data_for_fl_client(client_id: str, target_node_addresses: list[str], bc_connector, sybil_attack_log=None):
    print(f"FL Client {client_id}: Loading real data for nodes: {target_node_addresses}")
    client_features_list = []
    client_labels_list = []

    for node_addr in target_node_addresses:
        print(f"  Client {client_id}: Processing node {node_addr}")
        # Pass sybil_attack_log to generate_features_and_label_for_sybil
        features, label = generate_features_and_label_for_sybil(bc_connector, node_addr, sybil_attack_log=sybil_attack_log)
        client_features_list.append(features)
        client_labels_list.append(label)
        time.sleep(0.2) 

    if not client_features_list:
        return tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)

    features_array = np.array(client_features_list, dtype=np.float32)
    labels_array = np.array(client_labels_list, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((features_array, labels_array))
    return dataset.batch(1)

def make_federated_data_sybil_real(all_node_addresses, num_fl_clients=3, sybil_attack_log=None):
    # Load environment variables
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                           "w3storage-upload-script", "ifps_qr.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print(f"Warning: Environment file not found at {env_path}. Using default/placeholder values.")

    contract_address = os.getenv("CONTRACT_ADDRESS")
    rpc_url = os.getenv("POLYGON_AMOY_RPC")

    if not contract_address or not rpc_url:
        print("Warning: Contract address or RPC URL not found in environment variables")
        return []

    bc_connector = BlockchainConnector(rpc_url_override=rpc_url)
    
    if "SupplyChainNFT" not in bc_connector.contracts:
        print("Error: SupplyChainNFT contract not loaded...")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    if not all_node_addresses:
        print("Warning: No target node addresses provided...")
        empty_ds = tf.data.Dataset.from_tensor_slices((
            np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
            np.empty((0, 1), dtype=np.int32)
        )).batch(1)
        return [empty_ds for _ in range(num_fl_clients)]

    # Distribute all_node_addresses among clients
    nodes_per_client = len(all_node_addresses) // num_fl_clients
    extra_nodes = len(all_node_addresses) % num_fl_clients
    client_datasets = []
    current_node_idx = 0

    for i in range(num_fl_clients):
        client_id = f"real_data_client_{i}"
        num_nodes_for_this_client = nodes_per_client + (1 if i < extra_nodes else 0)
        client_node_list = []
        if num_nodes_for_this_client > 0 and current_node_idx < len(all_node_addresses):
            end_idx = min(current_node_idx + num_nodes_for_this_client, len(all_node_addresses))
            client_node_list = all_node_addresses[current_node_idx : end_idx]
            current_node_idx = end_idx
        
        if not client_node_list: 
            client_ds = tf.data.Dataset.from_tensor_slices((
                np.empty((0, NUM_SYBIL_FEATURES), dtype=np.float32),
                np.empty((0, 1), dtype=np.int32)
            )).batch(1)
        else:
            # Pass sybil_attack_log to load_real_data_for_fl_client
            client_ds = load_real_data_for_fl_client(client_id, client_node_list, bc_connector, sybil_attack_log=sybil_attack_log)
        client_datasets.append(client_ds)
    
    return client_datasets

if __name__ == '__main__':
    print("Testing Real Data Preparation for Sybil Detection with Heuristic and Log-based Labeling...")
    
    # Example: Load a dummy sybil_attack_log.json for testing if the main script isn't run first
    dummy_sybil_log_content = {
        "sybilNodes": [
            {
                "address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", 
                "initialization": [
                    {"action": "adminUpdateReputation", "newReputation": 10, "timestamp": "2024-01-01T10:00:00.000Z"},
                    {"action": "setNodeType", "nodeType": 1, "timestamp": "2024-01-01T10:00:05.000Z"}
                ],
                "activities": []
            }
        ],
        "scenarioA": {},
        "scenarioB": {},
        "scenarioC": {}
    }
    # Create a dummy log file for the test if it doesn't exist
    # In a real run, this comes from the CJS script
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                               "SupplyChain_dapp", "scripts", "lifecycle_demo")
    dummy_log_path = os.path.join(scripts_dir, "sybil_attack_log.json")
    
    sybil_log_for_test = None
    if os.path.exists(dummy_log_path):
        try:
            with open(dummy_log_path, "r") as f_log:
                sybil_log_for_test = json.load(f_log)
            print(f"Loaded existing sybil_attack_log.json from {dummy_log_path} for testing.")
        except Exception as e:
            print(f"Error loading existing sybil_attack_log.json: {e}. Using dummy log content.")
            sybil_log_for_test = dummy_sybil_log_content
    else:
        print(f"sybil_attack_log.json not found at {dummy_log_path}. Using dummy log content for testing.")
        # os.makedirs(scripts_dir, exist_ok=True) # Ensure directory exists if we were to write it
        # with open(dummy_log_path, "w") as f_dummy:
        #     json.dump(dummy_sybil_log_content, f_dummy, indent=2)
        # print(f"Created dummy sybil_attack_log.json at {dummy_log_path} for testing.")
        sybil_log_for_test = dummy_sybil_log_content

    example_node_addresses = [
        "0x5C6fF29A0f75E9d0dffC4374f600224EDc114449",  # Legitimate (e.g. contract deployer or a normal node)
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",  # Known Sybil from dummy log
        "0x1234567890123456789012345678901234567890",  # Unknown, will use heuristics
        "0xfedcba9876543210fedcba9876543210fedcba98"   # Unknown, will use heuristics
    ]

    print(f"Using example node addresses: {example_node_addresses}")

    num_clients_test = 2
    # Pass the loaded or dummy sybil_log_for_test to the make_federated_data function
    federated_train_data_test = make_federated_data_sybil_real(
        all_node_addresses=example_node_addresses, 
        num_fl_clients=num_clients_test,
        sybil_attack_log=sybil_log_for_test
    )
    
    print(f"\nCreated {len(federated_train_data_test)} client datasets.")
    for i, client_dataset in enumerate(federated_train_data_test):
        print(f"Client {i} dataset element spec: {client_dataset.element_spec}")
        num_elements = 0
        sybil_count = 0
        normal_count = 0
        for features_batch, labels_batch in client_dataset: 
            for j in range(features_batch.shape[0]): 
                num_elements += 1
                print(f"  Client {i}, Sample {num_elements}: Features: {features_batch[j].numpy()}, Label: {labels_batch[j].numpy()}")
                if labels_batch[j].numpy()[0] == 1:
                    sybil_count +=1
                else:
                    normal_count +=1
        if num_elements == 0: print(f"  Client {i} has no data.")
        print(f"  Client {i} total samples: {num_elements}. Sybil: {sybil_count}, Normal: {normal_count}")
    
    print("\nReal data preparation test for Sybil Detection with labeling complete.")

