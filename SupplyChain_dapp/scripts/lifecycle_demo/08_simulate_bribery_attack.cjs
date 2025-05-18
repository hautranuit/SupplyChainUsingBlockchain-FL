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
