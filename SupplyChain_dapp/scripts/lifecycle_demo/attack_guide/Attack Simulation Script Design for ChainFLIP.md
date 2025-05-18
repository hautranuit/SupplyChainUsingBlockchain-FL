# Attack Simulation Script Design for ChainFLIP

## Overview

This document outlines the design of two attack simulation scripts for the ChainFLIP blockchain system:
1. A Sybil attack simulation script
2. A Bribery attack simulation script

These scripts will be designed to integrate with the existing lifecycle demo and FL detection system.

## 1. Sybil Attack Simulation Script

### Filename: `07_simulate_sybil_attack.cjs`

### Purpose
Create multiple nodes controlled by the same entity and establish coordinated behavior patterns that can be detected by the FL models.

### Script Structure

```javascript
// Sybil Attack Simulation for ChainFLIP
const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');

// Configuration
const NUM_SYBIL_NODES = 4;  // Number of Sybil nodes to create
const ATTACK_INTENSITY = 0.8;  // 0.0 to 1.0, controls how obvious the attack is
const SYBIL_CONTROLLER = 'SybilMaster';  // Identifier for the controlling entity

async function main() {
  console.log("\n=== SIMULATING SYBIL ATTACK ===");
  
  // Load context
  const contextPath = path.join(__dirname, 'demo_context.json');
  const context = JSON.parse(fs.readFileSync(contextPath, 'utf8'));
  
  // Load contract
  const SupplyChainNFT = await ethers.getContractFactory("SupplyChainNFT");
  const contract = await SupplyChainNFT.attach(context.contractAddress);
  
  // Get signers for Sybil nodes
  const signers = await ethers.getSigners();
  const sybilSigners = signers.slice(10, 10 + NUM_SYBIL_NODES); // Use unused signers
  
  // Create Sybil node network
  const sybilNodes = [];
  console.log(`Creating ${NUM_SYBIL_NODES} Sybil nodes controlled by ${SYBIL_CONTROLLER}...`);
  
  for (let i = 0; i < NUM_SYBIL_NODES; i++) {
    const signer = sybilSigners[i];
    const nodeAddress = await signer.getAddress();
    
    // Register as verified node
    await contract.connect(signers[0]).setNodeVerificationStatus(nodeAddress, true);
    
    // Assign role (mix of transporters and retailers)
    const role = i % 2 === 0 ? 1 : 3; // Alternate between Transporter (1) and Retailer (3)
    await contract.connect(signers[0]).setNodeRole(nodeAddress, role);
    
    // Set initial reputation
    await contract.connect(signers[0]).setNodeReputation(nodeAddress, 80 + Math.floor(Math.random() * 20));
    
    // Add to Sybil network
    sybilNodes.push({
      address: nodeAddress,
      role: role === 1 ? 'Transporter' : 'Retailer',
      controller: SYBIL_CONTROLLER,
      sybilIndex: i
    });
    
    console.log(`  - Created Sybil node ${i+1}: ${nodeAddress} (${role === 1 ? 'Transporter' : 'Retailer'})`);
  }
  
  // Implement coordinated behavior
  console.log("\nImplementing coordinated Sybil behavior...");
  
  // 1. Similar transaction patterns
  if (context.productDetails && context.productDetails.length > 0) {
    // Coordinate around existing products
    for (let i = 0; i < Math.min(context.productDetails.length, 3); i++) {
      const product = context.productDetails[i];
      
      console.log(`  - Coordinating Sybil nodes around product ${product.tokenId}...`);
      
      // Create coordinated endorsements
      for (let j = 0; j < sybilNodes.length; j++) {
        const sybilSigner = sybilSigners[j];
        
        // Slight delay between transactions to avoid exact same block
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Endorse product quality (if contract has this function)
        try {
          await contract.connect(sybilSigner).endorseProduct(product.tokenId, 5); // Max rating
          console.log(`    - Node ${sybilNodes[j].address} endorsed product ${product.tokenId}`);
        } catch (error) {
          console.log(`    - Endorsement function not available, simulating endorsement in metadata`);
        }
      }
    }
  }
  
  // 2. Coordinated validation activities
  console.log("\n  - Implementing coordinated validation activities...");
  
  // Simulate batch validations with suspicious timing
  const batchIds = ["BATCH_" + Math.floor(Math.random() * 1000), "BATCH_" + Math.floor(Math.random() * 1000)];
  
  for (const batchId of batchIds) {
    console.log(`    - Coordinating validation for batch ${batchId}...`);
    
    // All Sybil nodes validate the same batch with very similar timestamps
    for (let i = 0; i < sybilNodes.length; i++) {
      // Record validation in metadata (actual contract call would be here if available)
      sybilNodes[i].validatedBatches = sybilNodes[i].validatedBatches || [];
      sybilNodes[i].validatedBatches.push({
        batchId: batchId,
        timestamp: new Date().toISOString(),
        decision: "Approved"
      });
      
      console.log(`      - Node ${sybilNodes[i].address} validated batch ${batchId}`);
      await new Promise(resolve => setTimeout(resolve, 200)); // Very small delay between validations
    }
  }
  
  // 3. Suspicious voting patterns
  if (context.proposals) {
    console.log("\n  - Implementing suspicious voting patterns...");
    
    for (const proposal of context.proposals) {
      // All Sybil nodes vote the same way
      const voteDecision = Math.random() > 0.5 ? "Approve" : "Reject";
      
      for (let i = 0; i < sybilNodes.length; i++) {
        sybilNodes[i].votes = sybilNodes[i].votes || [];
        sybilNodes[i].votes.push({
          proposalId: proposal.id,
          decision: voteDecision,
          timestamp: new Date().toISOString()
        });
        
        console.log(`    - Node ${sybilNodes[i].address} voted ${voteDecision} on proposal ${proposal.id}`);
      }
    }
  } else {
    // Create fictional proposals if none exist
    console.log("\n  - Creating and voting on fictional proposals...");
    
    const proposalIds = ["PROP_" + Math.floor(Math.random() * 1000), "PROP_" + Math.floor(Math.random() * 1000)];
    context.proposals = [];
    
    for (const proposalId of proposalIds) {
      const proposal = {
        id: proposalId,
        description: "Fictional proposal for Sybil detection",
        creator: context.deployerAddress,
        status: "Active"
      };
      
      context.proposals.push(proposal);
      
      // All Sybil nodes vote the same way
      const voteDecision = Math.random() > 0.5 ? "Approve" : "Reject";
      
      for (let i = 0; i < sybilNodes.length; i++) {
        sybilNodes[i].votes = sybilNodes[i].votes || [];
        sybilNodes[i].votes.push({
          proposalId: proposal.id,
          decision: voteDecision,
          timestamp: new Date().toISOString()
        });
        
        console.log(`    - Node ${sybilNodes[i].address} voted ${voteDecision} on proposal ${proposal.id}`);
      }
    }
  }
  
  // Update context with Sybil information
  console.log("\nUpdating context with Sybil attack information...");
  
  // Add Sybil nodes to context
  for (let i = 0; i < sybilNodes.length; i++) {
    const roleKey = sybilNodes[i].role.toLowerCase() + (i+1) + "SybilAddress";
    context[roleKey] = sybilNodes[i].address;
  }
  
  // Add Sybil network metadata
  context.sybilAttackSimulated = true;
  context.sybilNetwork = {
    controller: SYBIL_CONTROLLER,
    nodes: sybilNodes,
    attackIntensity: ATTACK_INTENSITY,
    simulationTimestamp: new Date().toISOString()
  };
  
  // Save updated context
  fs.writeFileSync(contextPath, JSON.stringify(context, null, 2));
  console.log(`Context updated with Sybil attack information at ${contextPath}`);
  
  // Create ground truth file for validation
  const groundTruthPath = path.join(__dirname, 'sybil_attack_ground_truth.json');
  const groundTruth = {
    attackType: "Sybil",
    controller: SYBIL_CONTROLLER,
    controlledNodes: sybilNodes.map(node => node.address),
    legitimateNodes: Object.entries(context)
      .filter(([key, value]) => key.endsWith('Address') && !key.includes('Sybil'))
      .map(([key, value]) => ({ role: key.replace('Address', ''), address: value })),
    simulationTimestamp: new Date().toISOString()
  };
  
  fs.writeFileSync(groundTruthPath, JSON.stringify(groundTruth, null, 2));
  console.log(`Ground truth data saved to ${groundTruthPath}`);
  
  console.log("\n=== SYBIL ATTACK SIMULATION COMPLETE ===");
}

// Execute the script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

## 2. Bribery Attack Simulation Script

### Filename: `08_simulate_bribery_attack.cjs`

### Purpose
Simulate bribery transactions and resulting behavioral changes in legitimate nodes to test FL detection capabilities.

### Script Structure

```javascript
// Bribery Attack Simulation for ChainFLIP
const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');

// Configuration
const NUM_BRIBED_NODES = 3;  // Number of nodes to bribe
const BRIBE_INTENSITY = 0.7;  // 0.0 to 1.0, controls how obvious the bribes are
const BRIBER_IDENTIFIER = 'MaliciousBriber';  // Identifier for the bribing entity

async function main() {
  console.log("\n=== SIMULATING BRIBERY ATTACK ===");
  
  // Load context
  const contextPath = path.join(__dirname, 'demo_context.json');
  const context = JSON.parse(fs.readFileSync(contextPath, 'utf8'));
  
  // Load contract
  const SupplyChainNFT = await ethers.getContractFactory("SupplyChainNFT");
  const contract = await SupplyChainNFT.attach(context.contractAddress);
  
  // Get signers
  const signers = await ethers.getSigners();
  const bribingEntity = signers[9]; // Use a dedicated signer as the briber
  
  // Identify target nodes to bribe (focus on validators, arbitrators, transporters)
  const potentialTargets = [];
  
  // Add arbitrator if exists
  if (context.arbitratorAddress) {
    potentialTargets.push({
      role: 'Arbitrator',
      address: context.arbitratorAddress,
      signerIndex: -1 // Will find the matching signer
    });
  }
  
  // Add transporters
  for (let i = 1; i <= 3; i++) {
    const key = `transporter${i}Address`;
    if (context[key]) {
      potentialTargets.push({
        role: `Transporter${i}`,
        address: context[key],
        signerIndex: -1
      });
    }
  }
  
  // Add retailer
  if (context.retailerAddress) {
    potentialTargets.push({
      role: 'Retailer',
      address: context.retailerAddress,
      signerIndex: -1
    });
  }
  
  // Find matching signers for each target
  for (let i = 0; i < potentialTargets.length; i++) {
    for (let j = 0; j < signers.length; j++) {
      if (await signers[j].getAddress() === potentialTargets[i].address) {
        potentialTargets[i].signerIndex = j;
        break;
      }
    }
  }
  
  // Filter out targets without matching signers
  const validTargets = potentialTargets.filter(target => target.signerIndex !== -1);
  
  // Select nodes to bribe
  const nodesToBribe = validTargets.slice(0, Math.min(NUM_BRIBED_NODES, validTargets.length));
  
  if (nodesToBribe.length === 0) {
    console.log("No valid targets found for bribery. Aborting simulation.");
    return;
  }
  
  console.log(`Selected ${nodesToBribe.length} nodes for bribery:`);
  nodesToBribe.forEach(node => {
    console.log(`  - ${node.role} (${node.address})`);
  });
  
  // Execute bribery transactions
  console.log("\nExecuting bribery transactions...");
  
  const briberyTransactions = [];
  const bribedNodes = [];
  
  for (const node of nodesToBribe) {
    // Calculate bribe amount (make it look like a normal transaction but with specific patterns)
    // Use wei amounts that look innocent but have specific patterns
    const baseAmount = ethers.utils.parseEther("0.01"); // 0.01 ETH base
    const randomFactor = Math.floor(Math.random() * 100) / 100; // 0.00 to 0.99
    const bribeAmount = baseAmount.add(ethers.utils.parseEther(randomFactor.toFixed(2)));
    
    console.log(`  - Bribing ${node.role} with ${ethers.utils.formatEther(bribeAmount)} ETH...`);
    
    try {
      // Send the bribe transaction
      const tx = await bribingEntity.sendTransaction({
        to: node.address,
        value: bribeAmount
      });
      
      await tx.wait();
      
      // Record the bribery transaction
      briberyTransactions.push({
        from: await bribingEntity.getAddress(),
        to: node.address,
        amount: ethers.utils.formatEther(bribeAmount),
        txHash: tx.hash,
        timestamp: new Date().toISOString()
      });
      
      // Add to bribed nodes list
      bribedNodes.push({
        role: node.role,
        address: node.address,
        bribeAmount: ethers.utils.formatEther(bribeAmount),
        bribeTimestamp: new Date().toISOString()
      });
      
      console.log(`    - Transaction successful: ${tx.hash}`);
    } catch (error) {
      console.log(`    - Transaction failed: ${error.message}`);
    }
    
    // Add small delay between transactions
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  // Simulate behavioral changes in bribed nodes
  console.log("\nSimulating behavioral changes in bribed nodes...");
  
  // 1. Manipulate dispute resolutions if arbitrator was bribed
  const bribedArbitrator = bribedNodes.find(node => node.role === 'Arbitrator');
  if (bribedArbitrator) {
    console.log(`  - Simulating biased dispute resolutions by arbitrator ${bribedArbitrator.address}...`);
    
    // Create or modify disputes to show bias
    const disputes = context.disputes || [];
    const newDisputes = [];
    
    // Create biased dispute resolutions
    for (let i = 0; i < 3; i++) {
      const disputeId = `DISPUTE_${Math.floor(Math.random() * 1000)}`;
      const favoredParty = await bribingEntity.getAddress(); // Briber is always favored
      
      newDisputes.push({
        id: disputeId,
        plaintiff: i % 2 === 0 ? favoredParty : context.manufacturerAddress,
        defendant: i % 2 === 0 ? context.manufacturerAddress : favoredParty,
        arbitrator: bribedArbitrator.address,
        outcome: "FavorPlaintiff", // Always favor the plaintiff (who is sometimes the briber)
        timestamp: new Date(Date.now() + i * 60000).toISOString(), // Staggered timestamps
        suspicious: true
      });
      
      console.log(`    - Created biased dispute resolution ${disputeId}`);
    }
    
    // Update context with new disputes
    context.disputes = [...disputes, ...newDisputes];
  }
  
  // 2. Manipulate batch validations if transporters were bribed
  const bribedTransporters = bribedNodes.filter(node => node.role.startsWith('Transporter'));
  if (bribedTransporters.length > 0) {
    console.log(`  - Simulating suspicious batch validations by ${bribedTransporters.length} transporters...`);
    
    // Create suspicious batch validations
    const batchValidations = context.batchValidations || [];
    const newValidations = [];
    
    for (const transporter of bribedTransporters) {
      // Create validations that always approve batches from the briber
      for (let i = 0; i < 2; i++) {
        const batchId = `BATCH_BRIBER_${Math.floor(Math.random() * 1000)}`;
        
        newValidations.push({
          batchId: batchId,
          validator: transporter.address,
          producer: await bribingEntity.getAddress(),
          decision: "Approved",
          quality: 5, // Maximum quality
          timestamp: new Date(Date.now() + i * 30000).toISOString(),
          suspicious: true
        });
        
        console.log(`    - Created suspicious batch validation ${batchId} by ${transporter.role}`);
      }
    }
    
    // Update context with new validations
    context.batchValidations = [...batchValidations, ...newValidations];
  }
  
  // 3. Manipulate product endorsements if retailer was bribed
  const bribedRetailer = bribedNodes.find(node => node.role === 'Retailer');
  if (bribedRetailer) {
    console.log(`  - Simulating biased product endorsements by retailer ${bribedRetailer.address}...`);
    
    // Create biased endorsements for products
    const endorsements = context.productEndorsements || [];
    const newEndorsements = [];
    
    // Find products to endorse
    const products = context.productDetails || [];
    
    for (const product of products) {
      // Create endorsement with maximum rating
      newEndorsements.push({
        productId: product.tokenId,
        endorser: bribedRetailer.address,
        rating: 5, // Maximum rating
        comment: "Excellent product, highly recommended!",
        timestamp: new Date().toISOString(),
        suspicious: true
      });
      
      console.log(`    - Created biased endorsement for product ${product.tokenId}`);
    }
    
    // Update context with new endorsements
    context.productEndorsements = [...endorsements, ...newEndorsements];
  }
  
  // Update context with bribery information
  console.log("\nUpdating context with bribery attack information...");
  
  // Add bribery metadata
  context.briberyAttackSimulated = true;
  context.briberyNetwork = {
    briber: await bribingEntity.getAddress(),
    briberIdentifier: BRIBER_IDENTIFIER,
    bribedNodes: bribedNodes,
    transactions: briberyTransactions,
    attackIntensity: BRIBE_INTENSITY,
    simulationTimestamp: new Date().toISOString()
  };
  
  // Save updated context
  fs.writeFileSync(contextPath, JSON.stringify(context, null, 2));
  console.log(`Context updated with bribery attack information at ${contextPath}`);
  
  // Create ground truth file for validation
  const groundTruthPath = path.join(__dirname, 'bribery_attack_ground_truth.json');
  const groundTruth = {
    attackType: "Bribery",
    briber: await bribingEntity.getAddress(),
    briberIdentifier: BRIBER_IDENTIFIER,
    bribedNodes: bribedNodes.map(node => node.address),
    legitimateNodes: Object.entries(context)
      .filter(([key, value]) => key.endsWith('Address') && 
              !bribedNodes.some(node => node.address === value))
      .map(([key, value]) => ({ role: key.replace('Address', ''), address: value })),
    simulationTimestamp: new Date().toISOString()
  };
  
  fs.writeFileSync(groundTruthPath, JSON.stringify(groundTruth, null, 2));
  console.log(`Ground truth data saved to ${groundTruthPath}`);
  
  console.log("\n=== BRIBERY ATTACK SIMULATION COMPLETE ===");
}

// Execute the script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

## 3. Integration with Orchestration Script

### Updates to `run_integrated_system.js`

Add the following code to the master orchestration script to enable attack simulations:

```javascript
// In the main execution flow section, after Phase 2 (Product Creation)
logMessage("\n=== OPTIONAL: ATTACK SIMULATION ===");
const simulateAttacks = process.env.SIMULATE_ATTACKS === 'true' || false;
const attackTypes = (process.env.ATTACK_TYPES || 'sybil,bribery').toLowerCase().split(',');

if (simulateAttacks) {
  logMessage(`Simulating attacks: ${attackTypes.join(', ')}`);
  
  // Simulate Sybil attack if requested
  if (attackTypes.includes('sybil')) {
    if (fs.existsSync(path.join(LIFECYCLE_DEMO_DIR, '07_simulate_sybil_attack.cjs'))) {
      runLifecycleScript('07_simulate_sybil_attack.cjs');
      
      // Run Sybil Detection after attack simulation
      if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_sybil_detection.py'))) {
        runFLModel('run_sybil_detection.py', 'Sybil Detection (Post-Attack)');
      }
    } else {
      logMessage("Sybil attack simulation script not found. Skipping.");
    }
  }
  
  // Simulate Bribery attack if requested
  if (attackTypes.includes('bribery')) {
    if (fs.existsSync(path.join(LIFECYCLE_DEMO_DIR, '08_simulate_bribery_attack.cjs'))) {
      runLifecycleScript('08_simulate_bribery_attack.cjs');
      
      // Run Node Behavior analysis after attack simulation
      if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
        runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Analysis (Post-Attack)');
      }
    } else {
      logMessage("Bribery attack simulation script not found. Skipping.");
    }
  }
}
```

## 4. Attack Validation Script

### Filename: `validate_attack_detection.py`

### Purpose
Validate the effectiveness of FL models in detecting the simulated attacks.

### Script Structure

```python
#!/usr/bin/env python3
import os
import sys
import json
import time
from datetime import datetime

def log_message(message):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attack_validation.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_message(f"Error loading {file_path}: {e}")
        return None

def main():
    log_message("=== Starting Attack Detection Validation ===")
    
    # Find ground truth files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lifecycle_demo_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 
                                     "SupplyChain_dapp/scripts/lifecycle_demo")
    
    sybil_ground_truth_path = os.path.join(lifecycle_demo_dir, "sybil_attack_ground_truth.json")
    bribery_ground_truth_path = os.path.join(lifecycle_demo_dir, "bribery_attack_ground_truth.json")
    
    # Find FL results
    results_dir = os.path.join(script_dir, "results")
    sybil_results_path = os.path.join(results_dir, "sybil_detection_results.json")
    node_behavior_results_path = os.path.join(results_dir, "node_behavior_timeseries_results.json")
    
    # Load ground truth data
    sybil_ground_truth = load_json_file(sybil_ground_truth_path)
    bribery_ground_truth = load_json_file(bribery_ground_truth_path)
    
    # Load FL results
    sybil_results = load_json_file(sybil_results_path)
    node_behavior_results = load_json_file(node_behavior_results_path)
    
    # Validate Sybil attack detection
    if sybil_ground_truth and sybil_results:
        log_message("\n--- Validating Sybil Attack Detection ---")
        
        # Extract ground truth Sybil nodes
        ground_truth_sybil_nodes = set(sybil_ground_truth.get('controlledNodes', []))
        log_message(f"Ground truth contains {len(ground_truth_sybil_nodes)} Sybil nodes")
        
        # Extract detected Sybil nodes (if available in results)
        detected_sybil_nodes = set()
        if 'detected_sybil_nodes' in sybil_results:
            detected_sybil_nodes = set(sybil_results['detected_sybil_nodes'])
            log_message(f"FL model detected {len(detected_sybil_nodes)} Sybil nodes")
        else:
            log_message("FL model results do not contain explicit Sybil detection outcomes")
            log_message("This is expected if the model is still in training phase")
        
        # Calculate detection metrics if available
        if detected_sybil_nodes:
            true_positives = len(ground_truth_sybil_nodes.intersection(detected_sybil_nodes))
            false_positives = len(detected_sybil_nodes - ground_truth_sybil_nodes)
            false_negatives = len(ground_truth_sybil_nodes - detected_sybil_nodes)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            log_message(f"Detection Metrics:")
            log_message(f"  - True Positives: {true_positives}")
            log_message(f"  - False Positives: {false_positives}")
            log_message(f"  - False Negatives: {false_negatives}")
            log_message(f"  - Precision: {precision:.2f}")
            log_message(f"  - Recall: {recall:.2f}")
            log_message(f"  - F1 Score: {f1_score:.2f}")
    else:
        log_message("Skipping Sybil attack validation - missing ground truth or results")
    
    # Validate Bribery attack detection
    if bribery_ground_truth and node_behavior_results:
        log_message("\n--- Validating Bribery Attack Detection ---")
        
        # Extract ground truth bribed nodes
        ground_truth_bribed_nodes = set(bribery_ground_truth.get('bribedNodes', []))
        log_message(f"Ground truth contains {len(ground_truth_bribed_nodes)} bribed nodes")
        
        # Extract detected anomalous nodes (if available in results)
        detected_anomalous_nodes = set()
        if 'detected_anomalous_nodes' in node_behavior_results:
            detected_anomalous_nodes = set(node_behavior_results['detected_anomalous_nodes'])
            log_message(f"FL model detected {len(detected_anomalous_nodes)} anomalous nodes")
        else:
            log_message("FL model results do not contain explicit anomaly detection outcomes")
            log_message("This is expected if the model is still in training phase")
        
        # Calculate detection metrics if available
        if detected_anomalous_nodes:
            true_positives = len(ground_truth_bribed_nodes.intersection(detected_anomalous_nodes))
            false_positives = len(detected_anomalous_nodes - ground_truth_bribed_nodes)
            false_negatives = len(ground_truth_bribed_nodes - detected_anomalous_nodes)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            log_message(f"Detection Metrics:")
            log_message(f"  - True Positives: {true_positives}")
            log_message(f"  - False Positives: {false_positives}")
            log_message(f"  - False Negatives: {false_negatives}")
            log_message(f"  - Precision: {precision:.2f}")
            log_message(f"  - Recall: {recall:.2f}")
            log_message(f"  - F1 Score: {f1_score:.2f}")
    else:
        log_message("Skipping Bribery attack validation - missing ground truth or results")
    
    # Save validation summary
    validation_summary = {
        "timestamp": datetime.now().isoformat(),
        "sybil_attack_validated": sybil_ground_truth is not None and sybil_results is not None,
        "bribery_attack_validated": bribery_ground_truth is not None and node_behavior_results is not None,
        "validation_complete": True
    }
    
    summary_path = os.path.join(results_dir, "attack_validation_summary.json")
    with open(summary_path, "w", encoding='utf-8') as f:
        json.dump(validation_summary, f, indent=2)
    
    log_message(f"Validation summary saved to {summary_path}")
    log_message("=== Attack Detection Validation Complete ===")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

## 5. Usage Instructions

To use the attack simulation scripts:

1. **Copy the scripts to the lifecycle_demo directory**:
   ```bash
   cp 07_simulate_sybil_attack.cjs 08_simulate_bribery_attack.cjs /path/to/SupplyChain_dapp/scripts/lifecycle_demo/
   ```

2. **Run the scripts individually after Phase 2**:
   ```bash
   cd /path/to/SupplyChain_dapp/scripts/lifecycle_demo
   node 07_simulate_sybil_attack.cjs
   node 08_simulate_bribery_attack.cjs
   ```

3. **Or run with the integrated system**:
   ```bash
   # Set environment variables to enable attack simulation
   export SIMULATE_ATTACKS=true
   export ATTACK_TYPES=sybil,bribery  # or just one: sybil or bribery
   
   # Run the integrated system
   cd /path/to/project/root
   node Federated\ Learning/run_integrated_system.js
   ```

4. **Validate attack detection**:
   ```bash
   cd /path/to/project/root/Federated\ Learning
   python fl_integration/validate_attack_detection.py
   ```

## 6. Next Steps

1. Implement the designed scripts
2. Test with the existing FL models
3. Refine attack patterns based on detection results
4. Update documentation with usage instructions
