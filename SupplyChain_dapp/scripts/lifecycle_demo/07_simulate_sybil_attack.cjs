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
