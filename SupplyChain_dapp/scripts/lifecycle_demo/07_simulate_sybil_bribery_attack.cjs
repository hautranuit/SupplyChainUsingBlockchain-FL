// Sybil Attack Simulation for ChainFLIP
const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');

// Configuration
const NUM_SYBIL_NODES = 3;  // Number of Sybil nodes to create (changed from 4 to 3)
// const ATTACK_INTENSITY = 0.8; // This will be handled by deterministic malicious actions
const SYBIL_CONTROLLER_DESCRIPTION = 'SybilMasterCoordinator'; // For logging/description

// Configuration for Bribery Attack (Scenario D)
const NUM_BRIBED_NODES = 2; // Number of nodes to attempt to bribe
const BRIBER_IDENTIFIER = 'MaliciousBriberScenarioD'; // Identifier for the bribing entity in Scenario D

// Helper function for delays (still used by the queue processor)
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// --- Transaction Queue System (copied from 06_scenario_dispute_resolution.cjs) ---
const transactionQueue = [];
let isProcessingQueue = false;
const MIN_INTERVAL_MS = 750; // Minimum interval between finishing one tx and starting the next

async function processTransactionQueue() {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    while (transactionQueue.length > 0) {
        const task = transactionQueue.shift();
        console.log(`  [TX_QUEUE] Executing: ${task.description}`);
        try {
            const txResponse = await task.action();
            if (txResponse && typeof txResponse.wait === 'function') {
                const receipt = await txResponse.wait();
                console.log(`    Transaction successful: ${receipt.hash}`);
                if (task.callback) {
                    task.callback(null, receipt);
                }
            } else {
                // For actions that don't return a typical tx response (e.g., view calls or already resolved promises)
                console.log(`    Action completed (no transaction receipt).`);
                if (task.callback) {
                    task.callback(null, txResponse);
                }
            }
        } catch (error) {
            console.error(`    Error executing ${task.description}:`, error.message);
            if (task.callback) {
                task.callback(error, null);
            }
        }
        if (transactionQueue.length > 0) {
            await delay(MIN_INTERVAL_MS);
        }
    }
    isProcessingQueue = false;
}

function addToTransactionQueue(description, action, callback) {
    return new Promise((resolve, reject) => {
        const wrappedCallback = (error, result) => {
            if (error) reject(error);
            else resolve(result);
            if (callback) callback(error, result); // Also call original callback if provided
        };
        transactionQueue.push({ description, action, callback: wrappedCallback });
        if (!isProcessingQueue) {
            processTransactionQueue().catch(err => {
                console.error("Error in processTransactionQueue:", err);
                // If the queue processing itself fails, we should probably reject pending promises.
                // For simplicity, current model relies on individual task errors being handled.
            });
        }
    });
}
// --- End Transaction Queue System ---

// Contract Enums (mirroring NodeManagement.sol)
const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4, Unassigned: 5 };
const ContractNodeType = { Primary: 0, Secondary: 1, Unspecified: 2 };
const BatchVoteType = { NotVoted: 0, Approve: 1, Deny: 2 };

// --- BEGIN NEW FUNCTION for Bribery Attack Simulation ---
async function simulateBriberyAttack(bribingEntitySigner, context, sybilAttackLog, signers, addToTransactionQueue, ethersInstance) {
    console.log(`  --- Starting Bribery Attack Simulation (Scenario D) ---`);
    sybilAttackLog.scenarioD.details.briber = bribingEntitySigner.address;
    sybilAttackLog.scenarioD.details.briberIdentifier = BRIBER_IDENTIFIER;

    const potentialTargets = [];
    if (context.arbitratorAddress) {
        potentialTargets.push({ role: 'Arbitrator', address: context.arbitratorAddress });
    }
    for (let i = 1; i <= 3; i++) {
        const key = `transporter${i}Address`;
        if (context[key]) {
            potentialTargets.push({ role: `Transporter${i}`, address: context[key] });
        }
    }
    if (context.retailerAddress) {
        potentialTargets.push({ role: 'Retailer', address: context.retailerAddress });
    }
    // Add other roles from context if they are potential bribe targets
    if (context.manufacturerAddress) {
        potentialTargets.push({ role: 'Manufacturer', address: context.manufacturerAddress });
    }
    if (context.customerAddress) {
        potentialTargets.push({ role: 'Customer', address: context.customerAddress });
    }


    const nodesToBribe = potentialTargets.filter(pt => pt.address !== bribingEntitySigner.address) // Cannot bribe self
                                     .slice(0, Math.min(NUM_BRIBED_NODES, potentialTargets.length));

    if (nodesToBribe.length === 0) {
        console.log("    No valid targets found for bribery (or NUM_BRIBED_NODES is 0). Skipping bribery actions.");
        sybilAttackLog.scenarioD.outcome = "No valid targets found or NUM_BRIBED_NODES is 0.";
        return;
    }

    console.log(`    Attempting to bribe ${nodesToBribe.length} nodes:`);
    nodesToBribe.forEach(node => console.log(`      - Target: ${node.role} (${node.address})`));

    let successfulBribes = 0;

    for (const node of nodesToBribe) {
        // Using bribe calculation logic similar to the original 08_simulate_bribery_attack.cjs
        const baseAmount = ethersInstance.utils.parseEther("0.01"); // 0.01 ETH base
        const randomFactor = Math.floor(Math.random() * 100) / 100; // 0.00 to 0.99
        const bribeAmount = baseAmount.add(ethersInstance.utils.parseEther(randomFactor.toFixed(2)));

        const bribeActionDescription = `Briber (${bribingEntitySigner.address}) sends bribe of ${ethersInstance.utils.formatEther(bribeAmount)} ETH to ${node.role} (${node.address})`;

        await addToTransactionQueue(
            bribeActionDescription,
            async () => bribingEntitySigner.sendTransaction({ to: node.address, value: bribeAmount }),
            (error, receipt) => {
                const bribeLogEntry = {
                    type: 'bribePayment',
                    briber: bribingEntitySigner.address,
                    targetRole: node.role,
                    targetAddress: node.address,
                    bribeAmountETH: ethersInstance.utils.formatEther(bribeAmount),
                    timestamp: new Date().toISOString(),
                };
                if (error) {
                    console.error(`      [BRIBERY_CALLBACK] Failed - ${bribeActionDescription}: ${error.message}`);
                    bribeLogEntry.status = 'failed';
                    bribeLogEntry.error = error.message;
                } else {
                    console.log(`      [BRIBERY_CALLBACK] Success - ${bribeActionDescription}. Tx: ${receipt.hash}`);
                    bribeLogEntry.status = 'success';
                    bribeLogEntry.txHash = receipt.hash;
                    sybilAttackLog.scenarioD.details.bribedNodes.push({
                        role: node.role,
                        address: node.address,
                        bribeAmountETH: ethersInstance.utils.formatEther(bribeAmount),
                        txHash: receipt.hash,
                        timestamp: bribeLogEntry.timestamp
                    });
                    successfulBribes++;
                }
                sybilAttackLog.scenarioD.actions.push(bribeLogEntry);
            }
        );
    }

    // Behavioral changes are logged after bribe transactions are processed by the queue.
    // For now, we'll add placeholder logs for expected behavioral changes.
    // The actual processing of these logs will happen when the queue is processed in main.

    console.log("    Queueing bribe payments. Simulated behavioral changes will be logged based on successful bribes.");
    // The outcome will be updated after processing the queue in main, based on actual successfulBribes count.
    // For now, set a preliminary outcome.
    sybilAttackLog.scenarioD.outcome = `Attempting to bribe ${nodesToBribe.length} nodes. Status of bribes will be updated after transaction processing.`;
    console.log(`  --- Bribery Attack Simulation Logic Queued ---`);
}

function logSimulatedBehavioralChanges(sybilAttackLog, bribingEntitySignerAddress) {
    console.log("    Logging simulated behavioral changes of successfully bribed nodes...");
    const bribedNodesInLog = sybilAttackLog.scenarioD.details.bribedNodes;

    if (bribedNodesInLog.length === 0) {
        console.log("      No nodes were successfully bribed, so no behavioral changes to log.");
        return;
    }

    const bribedArbitrator = bribedNodesInLog.find(n => n.role === 'Arbitrator');
    if (bribedArbitrator) {
        const simBehaviorLog = {
            type: "simulatedBehaviorChange",
            actorRole: "Arbitrator",
            actorAddress: bribedArbitrator.address,
            description: `Arbitrator ${bribedArbitrator.address} is considered bribed. Expected behavior: biased dispute resolutions favoring the briber (${bribingEntitySignerAddress}).`,
            timestamp: new Date().toISOString()
        };
        sybilAttackLog.scenarioD.actions.push(simBehaviorLog);
        console.log(`      - ${simBehaviorLog.description}`);
    }

    const bribedTransporters = bribedNodesInLog.filter(n => n.role.startsWith('Transporter'));
    if (bribedTransporters.length > 0) {
        bribedTransporters.forEach(transporter => {
            const simBehaviorLog = {
                type: "simulatedBehaviorChange",
                actorRole: transporter.role,
                actorAddress: transporter.address,
                description: `${transporter.role} ${transporter.address} is considered bribed. Expected behavior: approve suspicious batches, potentially collude in batch validation.`,
                timestamp: new Date().toISOString()
            };
            sybilAttackLog.scenarioD.actions.push(simBehaviorLog);
            console.log(`      - ${simBehaviorLog.description}`);
        });
    }
    
    const bribedRetailer = bribedNodesInLog.find(n => n.role === 'Retailer');
    if (bribedRetailer) {
        const simBehaviorLog = {
            type: "simulatedBehaviorChange",
            actorRole: "Retailer",
            actorAddress: bribedRetailer.address,
            description: `Retailer ${bribedRetailer.address} is considered bribed. Expected behavior: provide false/inflated endorsements for products associated with briber.`,
            timestamp: new Date().toISOString()
        };
        sybilAttackLog.scenarioD.actions.push(simBehaviorLog);
        console.log(`      - ${simBehaviorLog.description}`);
    }
}
// --- END NEW FUNCTION for Bribery Attack Simulation ---


async function main() {
    // --- Initial Setup ---
    console.log("=== INITIALIZING SIMULATION ENVIRONMENT ==="); 
    const signers = await ethers.getSigners();
    if (signers.length < 11) { // Base requirement for deployer + 3 sybils + other roles from context. Briber will use one of the 'other' available signers.
        throw new Error(`Need at least 11 signers. Found ${signers.length}. Deployer, ${NUM_SYBIL_NODES} Sybils, 1 Briber, and other roles defined in demo_context.json are required.`);
    }
    const deployerSigner = signers[0]; 
    const sybilSigners = signers.slice(8, 8 + NUM_SYBIL_NODES); 
    const briberSigner = signers[7]; // Assigning a specific signer for the briber. Ensure this doesn't clash.
                                     // signers[0] is deployer. signers[8,9,10] are sybils if NUM_SYBIL_NODES=3.
                                     // signers[1-6] might be used by roles in demo_context.json. signers[7] is chosen.

    // Define gas options
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('30', 'gwei'), 
        maxFeePerGas: ethers.parseUnits('100', 'gwei')       
    };
    console.log("Using gasOptions for transactions:", gasOptions);

    if (sybilSigners.length < NUM_SYBIL_NODES) {
        throw new Error(`Could not retrieve enough signers for Sybil nodes. Expected ${NUM_SYBIL_NODES}, got ${sybilSigners.length}. Total signers: ${signers.length}`);
    }
    console.log("Deployer Signer address:", deployerSigner.address);
    sybilSigners.forEach((signer, i) => console.log(`  Sybil Node ${i+1} (Signer Index ${8+i}) Address: ${signer.address}`));
    console.log(`Briber Signer (Scenario D) address: ${briberSigner.address}`);

    // Load context from demo_context.json
    const contextFilePath = path.join(__dirname, 'demo_context.json');
    if (!fs.existsSync(contextFilePath)) {
      console.error("Error: demo_context.json not found. Please run previous lifecycle scripts.");
      process.exit(1);
    }
    const context = JSON.parse(fs.readFileSync(contextFilePath, 'utf8'));
    const supplyChainNFTAddress = context.supplyChainNFTAddress || context.contractAddress; // Handle potential naming difference
    if (!supplyChainNFTAddress) {
      console.error("Error: supplyChainNFTAddress or contractAddress not found in demo_context.json.");
      process.exit(1);
    }
    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", supplyChainNFTAddress, deployerSigner);
    console.log(`Connected to SupplyChainNFT at ${supplyChainNFTAddress}`);

    // Initialize log object
    const sybilAttackLog = {
        simulationDate: new Date().toISOString(),
        simulationStartTime: new Date().toISOString(),        contractAddress: supplyChainNFTAddress,        sybilController: SYBIL_CONTROLLER_DESCRIPTION,
        numSybilNodes: NUM_SYBIL_NODES,
        sybilNodes: [],
        scenarioA: { description: "Sybil nodes attempt to gain high reputation rapidly through self-referrals or controlled interactions (simulated by admin setting high initial reputation).", actions: [], outcome: "" },
        scenarioB: { description: "Sybil nodes collude to dominate validation/arbitration processes (simulated by coordinated voting on a pre-existing batch).", actions: [], outcome: "" },
        scenarioC: { description: "Sybil nodes propose and approve a counterfeit/invalid batch of transactions.", actions: [], outcome: "" },
        scenarioD: {
            description: "A malicious entity (briber) attempts to influence network participants by sending them direct payments (bribes). Bribed nodes are expected to alter their behavior.",
            briberAddress: briberSigner.address,
            briberIdentifier: BRIBER_IDENTIFIER,
            numNodesTargetedForBribe: NUM_BRIBED_NODES,
            bribedNodes: [],   // Stores info about successfully bribed nodes { address, role, bribeAmountEth, txHash, timestamp }
            actions: [],       // Log of bribe transaction attempts
            simulatedBehavioralChanges: [], // Log of expected changes from bribed nodes
            outcome: "Bribery attempts logged. Behavioral changes are simulated for FL model training."
        }
    };

    console.log("\n=== SIMULATING SYBIL ATTACK ===");
    
    // REMOVED DUPLICATE DECLARATIONS that were here:
    // const contextFilePath = path.join(__dirname, 'demo_context.json');
    // const context = JSON.parse(fs.readFileSync(contextFilePath, 'utf8'));
    // const contractAddress = context.contractAddress;
    // const signers = await ethers.getSigners();
    // const deployerSigner = signers[0]; 
    // const sybilSigners = signers.slice(8, 8 + NUM_SYBIL_NODES); 
    // const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployerSigner);
    // const sybilAttackLog = { ... };

    // Ensure context is loaded using the already declared variables at the top of main
    if (!fs.existsSync(contextFilePath)) { // contextFilePath was declared at the top
      console.error("Error: demo_context.json not found. Please run previous lifecycle scripts.");
      process.exit(1);
    }
    // context was declared at the top
    // supplyChainNFTAddress was declared at the top
    // contractAddress is effectively supplyChainNFTAddress, ensure it's used consistently or aliased once.
    // For clarity, let's ensure we are using supplyChainNFTAddress for the contract object
    // and contractAddress for the log object if they must be different.
    // However, they should point to the same address.
    
    // The following lines re-confirm/re-log but don't re-declare, which is fine.
    // However, the actual re-declarations of signers, deployerSigner, sybilSigners, supplyChainNFT, sybilAttackLog
    // were the primary cause of the "Cannot redeclare" errors and have been removed.

    console.log(`Using SupplyChainNFT contract at: ${supplyChainNFTAddress}`);
    // signers, deployerSigner, sybilSigners were declared at the top
    
    console.log(`Deployer/Admin account: ${deployerSigner.address}`);
    console.log(`Sybil controller for this simulation: ${SYBIL_CONTROLLER_DESCRIPTION}`);
    console.log(`Found ${sybilSigners.length} Sybil signers (expected ${NUM_SYBIL_NODES}).`);
    sybilSigners.forEach((signer, i) => console.log(`  Sybil Node ${i+1} Address: ${signer.address}`));

    // supplyChainNFT was declared and initialized at the top using supplyChainNFTAddress.
    // sybilAttackLog was initialized at the top.

    console.log(`\n--- Initializing ${NUM_SYBIL_NODES} Sybil Nodes ---`);
    for (let i = 0; i < NUM_SYBIL_NODES; i++) {
      const sybilNodeSigner = sybilSigners[i];
      const sybilNodeAddress = sybilNodeSigner.address;
      // Initialize nodeLog structure for each Sybil node
      const nodeLog = { 
        address: sybilNodeAddress, 
        signerIndex: 8 + i, // For reference
        initialization: [], 
        activities: [] 
      };
      sybilAttackLog.sybilNodes.push(nodeLog);

      console.log(`  Initializing Sybil Node ${i+1} (${sybilNodeAddress})...`);

      await addToTransactionQueue(
          `Sybil ${i+1} (${sybilNodeAddress}): Set Verified Node`,
          () => supplyChainNFT.connect(deployerSigner).setVerifiedNode(sybilNodeAddress, true, gasOptions),
          (err, receipt) => {
              const logEntry = { 
                  action: 'setVerifiedNode', 
                  verified: true,
                  status: err ? 'failed' : 'success', 
                  txHash: receipt ? receipt.hash : null,
                  error: err ? err.message : null 
              };
              sybilAttackLog.sybilNodes[i].initialization.push(logEntry);
              if(err) console.error(`    [TX_QUEUE_CALLBACK] Failed - ${logEntry.action} for Sybil ${i+1}: ${err.message}`);
              else console.log(`    [TX_QUEUE_CALLBACK] Success - ${logEntry.action} for Sybil ${i+1}. Tx: ${logEntry.txHash}`);
          }
      );

      // Set Role
      const roleToSet = ContractRole.Retailer; // Sybils are Retailers
      await addToTransactionQueue(
          `Admin sets Role for Sybil Node ${i+1} (${sybilNodeAddress}) to ${Object.keys(ContractRole).find(key => ContractRole[key] === roleToSet)}`,
          () => supplyChainNFT.connect(deployerSigner).adminSetRole(sybilNodeAddress, roleToSet, gasOptions),
          (err, receipt) => { // Changed 'error' to 'err' and 'result' to 'receipt' for consistency
              const logEntry = {
                  action: 'adminSetRole',
                  role: Object.keys(ContractRole).find(key => ContractRole[key] === roleToSet),
                  status: err ? 'failed' : 'success',
                  txHash: receipt ? receipt.hash : null,
                  error: err ? err.message : null
              };
              sybilAttackLog.sybilNodes[i].initialization.push(logEntry);
              if(err) console.error(`    [TX_QUEUE_CALLBACK] Failed - ${logEntry.action} for Sybil ${i+1}: ${err.message}`);
              else console.log(`    [TX_QUEUE_CALLBACK] Success - ${logEntry.action} for Sybil ${i+1} to ${logEntry.role}. Tx: ${logEntry.txHash}`);
          }
      );

      // Set Node Type to Secondary (để có thể propose batch)
      const nodeTypeToSet = ContractNodeType.Secondary;
      await addToTransactionQueue(
          `Admin sets NodeType for Sybil Node ${i+1} (${sybilNodeAddress}) to ${Object.keys(ContractNodeType).find(key => ContractNodeType[key] === nodeTypeToSet)}`,
          () => supplyChainNFT.connect(deployerSigner).setNodeType(sybilNodeAddress, nodeTypeToSet, gasOptions), // Corrected: adminSetNodeType -> setNodeType
          (error, result) => { // 'result' here is the receipt
              const success = !error;
              const logEntry = {
                  action: 'setNodeType',
                  nodeType: Object.keys(ContractNodeType).find(key => ContractNodeType[key] === nodeTypeToSet),
                  status: success ? 'success' : 'failed',
                  txHash: result ? result.hash : null, // Assuming result might be a transaction receipt if successful
                  error: error ? error.message : null
              };
              sybilAttackLog.sybilNodes[i].initialization.push(logEntry);
              if(error) console.error(`    [TX_QUEUE_CALLBACK] Failed - ${logEntry.action} for Sybil ${i+1}: ${error.message}`);
              else console.log(`    [TX_QUEUE_CALLBACK] Success - ${logEntry.action} for Sybil ${i+1} to ${logEntry.nodeType}. Tx: ${logEntry.txHash}`);
          }
      );

      // Set Reputation
      const initialReputationForSybil = 75; 
      await addToTransactionQueue(
          `Sybil ${i+1} (${sybilNodeAddress}): Set Initial Reputation to ${initialReputationForSybil}`,
          () => supplyChainNFT.connect(deployerSigner).adminUpdateReputation(sybilNodeAddress, initialReputationForSybil, gasOptions),
          (err, receipt) => {
              const logEntry = {
                  action: 'adminUpdateReputation',
                  reputation: initialReputationForSybil,
                  status: err ? 'failed' : 'success',
                  txHash: receipt ? receipt.hash : null,
                  error: err ? err.message : null
              };
              sybilAttackLog.sybilNodes[i].initialization.push(logEntry);
              if(err) console.error(`    [TX_QUEUE_CALLBACK] Failed - ${logEntry.action} for Sybil ${i+1}: ${err.message}`);
              else console.log(`    [TX_QUEUE_CALLBACK] Success - ${logEntry.action} for Sybil ${i+1} to ${logEntry.reputation}. Tx: ${logEntry.txHash}`);
          }
      );
    }
    console.log("  Processing Sybil node initializations in queue...");
    await processTransactionQueue(); 
    console.log("--- Sybil Node Initialization Complete ---");

    console.log("\n--- Simulating Malicious Sybil Activities ---");

    let targetTokenId;
    if (context.productDetails && context.productDetails.length > 0 && context.productDetails[0].tokenId) {
      targetTokenId = context.productDetails[0].tokenId;
      console.log(`  Targeting product with Token ID: ${targetTokenId} for dispute simulation.`);
    } else {
      console.warn("  Warning: No product details found in demo_context.json. Dispute simulation will be skipped.");
      targetTokenId = null; 
    }

    if (targetTokenId) {
      console.log("  Scenario A: Sybils creating frivolous disputes...");
      for (let i = 0; i < NUM_SYBIL_NODES; i++) {
        const sybilNodeSigner = sybilSigners[i];
        const sybilNodeAddress = sybilNodeSigner.address;
        
        const currentTokenId = targetTokenId; 
        const currentReason = `Frivolous dispute by Sybil ${sybilNodeAddress} (Node ${i+1}) for token ${currentTokenId}`;
        const currentEvidenceCID = `fake_evidence_cid_sybil_${i}_${Date.now()}`; 

        await addToTransactionQueue(
          `Sybil ${i+1} (${sybilNodeAddress}): Open Frivolous Dispute for Token ${currentTokenId}`,
          () => supplyChainNFT.connect(sybilNodeSigner).openDispute(currentTokenId, currentReason, currentEvidenceCID, gasOptions),
          (err, receipt) => { 
            const activityEntryBase = { 
              type: 'openDispute', 
              actor: sybilNodeAddress, // Added for scenario log
              tokenId: currentTokenId, 
              reason: currentReason,
              evidenceCID: currentEvidenceCID,
              timestamp: new Date().toISOString() // Added for scenario log
            };
            if (err) {
              console.error(`    [TX_QUEUE_CALLBACK] Failed - Sybil ${i+1} (${sybilNodeAddress}) openDispute on token ${currentTokenId}: ${err.message}`);
              const failedEntry = { 
                  ...activityEntryBase,
                  status: 'failed', 
                  error: err.message 
              };
              sybilAttackLog.sybilNodes[i].activities.push(failedEntry);
              sybilAttackLog.scenarioA.actions.push(failedEntry); // Log to Scenario A
            } else {
              let disputeIdLogged = "unknown";
              if (receipt && receipt.events) {
                  const disputeOpenedEvent = receipt.events.find(e => e.event === 'DisputeOpened');
                  if (disputeOpenedEvent && disputeOpenedEvent.args && disputeOpenedEvent.args.disputeId !== undefined) {
                      disputeIdLogged = disputeOpenedEvent.args.disputeId.toString();
                  }
              }
              console.log(`    [TX_QUEUE_CALLBACK] Success - Sybil ${i+1} (${sybilNodeAddress}) openDispute on token ${currentTokenId}. Tx: ${receipt ? receipt.hash : 'N/A'}, DisputeID: ${disputeIdLogged}`);
              const successEntry = { 
                  ...activityEntryBase,
                  status: 'success', 
                  txHash: receipt ? receipt.hash : null,
                  disputeId: disputeIdLogged
              };
              sybilAttackLog.sybilNodes[i].activities.push(successEntry);
              sybilAttackLog.scenarioA.actions.push(successEntry); // Log to Scenario A
            }
          }
        );
      }
    }

    const targetBatchId = context.targetBatchIdForSybilTest; 
    if (targetBatchId) {
      console.log(`  Scenario B: Sybils attempting coordinated voting on Batch ID: ${targetBatchId}...`);
      // Pre-condition check: Ensure Sybils are validators for this batch type, or this will likely fail.
      // This script assumes they might have been made validators in a previous step or by an admin.
      // If not, the contract calls might revert. The log will capture this.

      for (let i = 0; i < NUM_SYBIL_NODES; i++) {
        const sybilNodeSigner = sybilSigners[i];
        const sybilNodeAddress = sybilNodeSigner.address;
        const voteDecision = BatchVoteType.Deny; // All Sybils vote to Deny

        await addToTransactionQueue(
          `Sybil ${i+1} (${sybilNodeAddress}): Vote ${Object.keys(BatchVoteType).find(k => BatchVoteType[k] === voteDecision)} on Batch ${targetBatchId}`,
          () => supplyChainNFT.connect(sybilNodeSigner).castBatchVote(targetBatchId, voteDecision, gasOptions), 
          (err, receipt) => { 
              const activityEntryBase = {
                  type: 'castBatchVote',
                  batchId: targetBatchId,
                  vote: Object.keys(BatchVoteType).find(k => BatchVoteType[k] === voteDecision),
                  actor: sybilNodeAddress, // Changed from nodeAddress for consistency
                  timestamp: new Date().toISOString() // Added for scenario log
              };
              if (err) {
                  console.error(`    [TX_QUEUE_CALLBACK] Failed - Sybil ${i+1} (${sybilNodeAddress}) castBatchVote on batch ${targetBatchId}: ${err.message}`);
                  const failedEntry = {
                      ...activityEntryBase,
                      status: 'failed',
                      error: err.message
                  };
                  sybilAttackLog.sybilNodes[i].activities.push(failedEntry);
                  sybilAttackLog.scenarioB.actions.push(failedEntry); // Log to Scenario B
              } else {
                  console.log(`    [TX_QUEUE_CALLBACK] Success - Sybil ${i+1} (${sybilNodeAddress}) castBatchVote on batch ${targetBatchId} as ${activityEntryBase.vote}. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                  const successEntry = {
                      ...activityEntryBase,
                      status: 'success',
                      txHash: receipt ? receipt.hash : null
                  };
                  sybilAttackLog.sybilNodes[i].activities.push(successEntry);
                  sybilAttackLog.scenarioB.actions.push(successEntry); // Log to Scenario B
              }
          }
        );
      }
    } else {
      sybilAttackLog.scenarioB.outcome = "Skipped: targetBatchIdForSybilTest not found in demo_context.json.";
      console.warn("  Warning: targetBatchIdForSybilTest not found in demo_context.json. Sybil collusion (Scenario B) will be skipped.");
    }
    
    console.log("  Processing Sybil malicious activities in queue...");
    await processTransactionQueue();
    console.log("--- Sybil Malicious Activities Simulation Complete ---");

    // Update outcomes for Scenario A and B
    if (targetTokenId) {
        const scenarioAAttempts = sybilAttackLog.scenarioA.actions.length;
        const scenarioASuccesses = sybilAttackLog.scenarioA.actions.filter(a => a.status === 'success').length;
        sybilAttackLog.scenarioA.outcome = `Scenario A: ${scenarioASuccesses}/${scenarioAAttempts} frivolous dispute attempts recorded. Target token ID: ${targetTokenId}.`;
    } else {
        sybilAttackLog.scenarioA.outcome = "Scenario A: Skipped due to no targetTokenId found in context.";
    }

    if (targetBatchId) {
        const scenarioBAttempts = sybilAttackLog.scenarioB.actions.length;
        const scenarioBSuccesses = sybilAttackLog.scenarioB.actions.filter(a => a.status === 'success').length;
        sybilAttackLog.scenarioB.outcome = `Scenario B: ${scenarioBSuccesses}/${scenarioBAttempts} collusive voting attempts recorded on batch ID: ${targetBatchId}.`;
    } else {
        sybilAttackLog.scenarioB.outcome = "Scenario B: Skipped due to no targetBatchIdForSybilTest found in context.";
    }

    // Kịch bản C: Sybil nodes đề xuất và phê duyệt một lô hàng giả mạo
    console.log('\\\\n--- Kịch bản C: Sybil Nodes Propose and Approve Counterfeit Batch ---');
    const sybilProposer = sybilSigners[0]; // Sybil đầu tiên sẽ đề xuất

    // Tạo dữ liệu giao dịch giả mạo
    // Giả sử chúng ta không có NFT thực tế để chuyển, chúng ta sẽ tạo dữ liệu giao dịch tượng trưng
    // Trong một kịch bản thực tế hơn, Sybil có thể mint một NFT "giả" nếu có quyền MINTER_ROLE
    // hoặc cố gắng chuyển một NFT không thuộc sở hữu của mình (sẽ bị revert nếu không có cơ chế chiếm quyền).
    // Vì mục đích mô phỏng tấn công vào logic batch, chúng ta tập trung vào việc batch được đề xuất và bỏ phiếu.
    const counterfeitTransactions = [
        {
            from: sybilProposer.address, // Giao dịch giả mạo từ chính Sybil
            to: ethers.Wallet.createRandom().address, // Đến một địa chỉ ngẫu nhiên
            tokenId: 999999, // ID token không tồn tại hoặc giả mạo
            // Thêm các trường khác nếu TransactionData yêu cầu
        },
        {
            from: sybilProposer.address,
            to: ethers.Wallet.createRandom().address,
            tokenId: 999998, 
        }
    ];

    let proposedBatchIdScenarioC;

    // Sybil node 0 đề xuất lô hàng giả mạo
    await addToTransactionQueue(
        `Sybil Node 0 (${sybilProposer.address}) proposes a counterfeit batch`,
        async () => {
            const tx = await supplyChainNFT.connect(sybilProposer).proposeBatch(counterfeitTransactions, gasOptions);
            const receipt = await tx.wait();
            // Tìm event BatchProposed để lấy batchId
            const event = receipt.events?.find(e => e.event === 'BatchProposed');
            if (event && event.args) {
                proposedBatchIdScenarioC = event.args.batchId;
                console.log(`  Counterfeit Batch Proposed with ID: ${proposedBatchIdScenarioC}`);
                sybilAttackLog.scenarioC.actions.push({
                    actor: sybilProposer.address,
                    action: 'ProposeCounterfeitBatch',
                    batchId: proposedBatchIdScenarioC.toNumber(),
                    transactions: counterfeitTransactions.map(t => ({...t, tokenId: t.tokenId.toString()})),
                    timestamp: new Date().toISOString(),
                    status: 'Success'
                });
            } else {
                throw new Error("BatchProposed event not found or batchId missing.");
            }
            return tx; // Trả về tx để processTransactionQueue có thể wait
        },
        (error) => {
            sybilAttackLog.scenarioC.actions.push({
                actor: sybilProposer.address,
                action: 'ProposeCounterfeitBatch',
                transactions: counterfeitTransactions.map(t => ({...t, tokenId: t.tokenId.toString()})),
                timestamp: new Date().toISOString(),
                status: 'Failed',
                error: error ? error.message : 'Unknown error'
            });
        }
    );
    await processTransactionQueue(); // Đảm bảo đề xuất được xử lý trước khi bỏ phiếu

    if (proposedBatchIdScenarioC === undefined) {
        console.error('  [SCENARIO C] Failed to propose counterfeit batch. Skipping votes.');
        sybilAttackLog.scenarioC.outcome = 'Failed to propose batch';
    } else {
        // Tất cả Sybil nodes (bao gồm cả người đề xuất) bỏ phiếu Approve cho lô hàng giả mạo
        for (let i = 0; i < sybilSigners.length; i++) {
            const currentSybil = sybilSigners[i];
            await addToTransactionQueue(
                `Sybil Node ${i} (${currentSybil.address}) votes to APPROVE counterfeit batch ${proposedBatchIdScenarioC}`,
                () => supplyChainNFT.connect(currentSybil).validateBatch(proposedBatchIdScenarioC, true, gasOptions), // true for Approve
                (error) => {
                    sybilAttackLog.scenarioC.actions.push({
                        actor: currentSybil.address,
                        action: 'VoteCounterfeitBatch',
                        batchId: proposedBatchIdScenarioC.toNumber(),
                        vote: 'Approve',
                        timestamp: new Date().toISOString(),
                        status: error ? 'Failed' : 'Success',
                        error: error ? error.message : undefined
                    });
                }
            );
        }
        await processTransactionQueue(); // Xử lý tất cả các phiếu bầu

        // Kiểm tra trạng thái của batch sau khi bỏ phiếu (tùy chọn, có thể cần thêm logic)
        // Ví dụ: gọi supplyChainNFT.getBatchDetails(proposedBatchIdScenarioC)
        // và ghi log kết quả vào sybilAttackLog.scenarioC.outcome
        try {
            const batchDetails = await supplyChainNFT.getBatchDetails(proposedBatchIdScenarioC);
            console.log(`  [SCENARIO C] Details of counterfeit batch ${proposedBatchIdScenarioC}:`, {
                validated: batchDetails.validated,
                committed: batchDetails.committed,
                approvals: batchDetails.approvals ? batchDetails.approvals.toString() : (await supplyChainNFT.batchApprovals(proposedBatchIdScenarioC)).toString(), // Fallback if not in struct
                denials: batchDetails.denials ? batchDetails.denials.toString() : (await supplyChainNFT.batchDenials(proposedBatchIdScenarioC)).toString(), // Fallback
                proposer: batchDetails.proposer,
                flagged: batchDetails.flagged
            });
            sybilAttackLog.scenarioC.outcome = {
                message: 'Sybil nodes attempted to approve a counterfeit batch.',
                batchId: proposedBatchIdScenarioC.toNumber(),
                details: {
                    validated: batchDetails.validated,
                    committed: batchDetails.committed,
                    approvals: (await supplyChainNFT.batchApprovals(proposedBatchIdScenarioC)).toString(),
                    denials: (await supplyChainNFT.batchDenials(proposedBatchIdScenarioC)).toString(),
                    proposer: batchDetails.proposer,
                    flagged: batchDetails.flagged
                }
            };
        } catch (e) {
            console.error(`  [SCENARIO C] Error fetching details for batch ${proposedBatchIdScenarioC}: ${e.message}`);
            sybilAttackLog.scenarioC.outcome = `Error fetching details for batch ${proposedBatchIdScenarioC}: ${e.message}`;
        }
    }

    // --- Scenario D: Bribery Attack ---
    console.log('\n--- Scenario D: Simulating Bribery Attack ---');
    await simulateBriberyAttack(context, signers, briberSigner, sybilAttackLog, addToTransactionQueue);
    
    console.log("  Processing bribery transactions in queue...");
    await processTransactionQueue(); // Process any queued bribery transactions
    
    // Log simulated behavioral changes based on successful bribes recorded in sybilAttackLog
    await logSimulatedBehavioralChanges(context, sybilAttackLog.scenarioD.bribedNodes, sybilAttackLog, briberSigner.address);
    console.log("--- Bribery Attack Simulation (Scenario D) Complete ---");


    sybilAttackLog.simulationEndTime = new Date().toISOString();
    const logFilePath = path.join(__dirname, 'sybil_attack_log.json');
    fs.writeFileSync(logFilePath, JSON.stringify(sybilAttackLog, null, 2));
    console.log(`\nSybil attack simulation log saved to: ${logFilePath}`);

    console.log("\n=== SYBIL ATTACK SIMULATION FINISHED ===");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Critical error in Sybil attack simulation script:", error);
        process.exit(1);
    });

// --- Bribery Attack Simulation Functions (Adapted from 08_simulate_bribery_attack.cjs) ---

async function simulateBriberyAttack(context, allSigners, briberSigner, sybilAttackLog, addToTransactionQueue) {
    console.log(`  [SCENARIO D] Initiating bribery simulation. Briber: ${briberSigner.address}`);

    const potentialTargets = [];
    // Gather potential targets from demo_context.json
    if (context.arbitratorAddress) {
        potentialTargets.push({ role: 'Arbitrator', address: context.arbitratorAddress });
    }
    for (let i = 1; i <= 3; i++) { // Assuming up to 3 transporters from context
        if (context[`transporter${i}Address`]) {
            potentialTargets.push({ role: `Transporter${i}`, address: context[`transporter${i}Address`] });
        }
    }
    if (context.retailerAddress) {
        potentialTargets.push({ role: 'Retailer', address: context.retailerAddress });
    }
    // Add other roles if necessary, e.g., manufacturer
    if (context.manufacturerAddress) {
        potentialTargets.push({ role: 'Manufacturer', address: context.manufacturerAddress });
    }

    // Avoid bribing the briber itself or the deployer/admin
    const validTargets = potentialTargets.filter(target => 
        target.address.toLowerCase() !== briberSigner.address.toLowerCase() &&
        target.address.toLowerCase() !== allSigners[0].address.toLowerCase() // allSigners[0] is deployer
    );

    // Select a subset of valid targets to bribe
    const nodesToAttemptBribe = validTargets.sort(() => 0.5 - Math.random()).slice(0, NUM_BRIBED_NODES);

    if (nodesToAttemptBribe.length === 0) {
        console.log("  [SCENARIO D] No valid targets found for bribery based on context and configuration.");
        sybilAttackLog.scenarioD.outcome = "No valid targets found for bribery.";
        return;
    }

    console.log(`  [SCENARIO D] Selected ${nodesToAttemptBribe.length} nodes to attempt to bribe:`);
    nodesToAttemptBribe.forEach(node => console.log(`    - Role: ${node.role}, Address: ${node.address}`));

    for (const targetNode of nodesToAttemptBribe) {
        // Calculate a somewhat "realistic" bribe amount in Ether
        const baseAmountEth = 0.01; // Base 0.01 ETH
        const randomFactor = Math.floor(Math.random() * 5) / 100; // 0.00 to 0.04 ETH
        const bribeAmountWei = ethers.parseEther((baseAmountEth + randomFactor).toFixed(4));
        const bribeAmountEthFormatted = ethers.formatEther(bribeAmountWei);

        const description = `[SCENARIO D] Briber (${briberSigner.address}) sends ${bribeAmountEthFormatted} ETH to ${targetNode.role} (${targetNode.address})`;

        await addToTransactionQueue(
            description,
            () => briberSigner.sendTransaction({
                to: targetNode.address,
                value: bribeAmountWei
                // gasOptions are not typically used for direct ETH transfers via sendTransaction
            }),
            (error, receipt) => {
                const logEntry = {
                    type: 'bribePaymentAttempt',
                    briber: briberSigner.address,
                    targetAddress: targetNode.address,
                    targetRole: targetNode.role,
                    amountEth: bribeAmountEthFormatted,
                    timestamp: new Date().toISOString(),
                };
                if (error) {
                    console.error(`    [TX_QUEUE_CALLBACK] Failed - ${description}: ${error.message}`);
                    logEntry.status = 'failed';
                    logEntry.error = error.message;
                } else {
                    console.log(`    [TX_QUEUE_CALLBACK] Success - ${description}. Tx: ${receipt.hash}`);
                    logEntry.status = 'success';
                    logEntry.txHash = receipt.hash;
                    // Add to the list of successfully bribed nodes for behavioral simulation logging
                    sybilAttackLog.scenarioD.bribedNodes.push({
                        address: targetNode.address,
                        role: targetNode.role,
                        bribeAmountEth: bribeAmountEthFormatted,
                        txHash: receipt.hash,
                        timestamp: logEntry.timestamp
                    });
                }
                sybilAttackLog.scenarioD.actions.push(logEntry);
            }
        );
    }
    // Note: processTransactionQueue() will be called in main after this function returns
}

async function logSimulatedBehavioralChanges(context, successfullyBribedNodes, sybilAttackLog, briberAddress) {
    console.log("  [SCENARIO D] Logging simulated behavioral changes for bribed nodes...");
    if (!successfullyBribedNodes || successfullyBribedNodes.length === 0) {
        console.log("  [SCENARIO D] No nodes were successfully bribed, so no behavioral changes to log.");
        return;
    }

    for (const bribedNode of successfullyBribedNodes) {
        let simulatedChange = {
            type: 'simulatedBehavioralChange',
            actor: bribedNode.address,
            role: bribedNode.role,
            influenceSource: briberAddress,
            timestamp: new Date().toISOString(),
            details: []
        };

        switch (bribedNode.role) {
            case 'Arbitrator':
                simulatedChange.details.push("Expected to show bias in dispute resolutions, potentially favoring the briber or their associates.");
                simulatedChange.details.push("May attempt to delay or unfairly close disputes against briber's interests.");
                // Example: find a dispute from context and log an expected biased resolution
                if (context.disputes && context.disputes.length > 0) {
                    const targetDispute = context.disputes[0]; // Simplistic choice
                    simulatedChange.details.push(`For instance, for a dispute like ID ${targetDispute.id || 'unknown_dispute_id'}, might rule in favor of party 'X' due to bribe.`);
                }
                break;
            case 'Transporter1':
            case 'Transporter2':
            case 'Transporter3':
                simulatedChange.details.push("Expected to falsely validate batches, e.g., approve counterfeit/low-quality goods or attest to incorrect transport conditions.");
                simulatedChange.details.push("May collude with briber to bypass quality checks for specific batches.");
                 if (context.targetBatchIdForSybilTest) { // or another relevant batchId
                    simulatedChange.details.push(`Could falsely approve Batch ID ${context.targetBatchIdForSybilTest} if beneficial to briber.`);
                }
                break;
            case 'Retailer':
                simulatedChange.details.push("Expected to provide false endorsements or positive reviews for substandard products associated with the briber.");
                simulatedChange.details.push("May hide negative information or complaints about certain products.");
                if (context.productDetails && context.productDetails.length > 0) {
                    const targetProduct = context.productDetails[0];
                     simulatedChange.details.push(`Might give a fake 5-star review to Product Token ID ${targetProduct.tokenId}.`);
                }
                break;
            case 'Manufacturer':
                simulatedChange.details.push("Expected to potentially approve substandard raw materials or components from suppliers linked to the briber.");
                simulatedChange.details.push("May falsify production records or quality control data for certain batches if it benefits the briber.");
                break;
            default:
                simulatedChange.details.push("Generic behavioral change: Expected to act in favor of the briber's interests, potentially violating protocol rules or ethical standards.");
                break;
        }
        sybilAttackLog.scenarioD.simulatedBehavioralChanges.push(simulatedChange);
        console.log(`    - Logged expected changes for ${bribedNode.role} (${bribedNode.address})`);
    }
}
