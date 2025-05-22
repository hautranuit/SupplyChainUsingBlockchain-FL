// Improved Sybil Attack Simulation for ChainFLIP
const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');
const { DateTime } = require('luxon'); // Add luxon for better timestamp handling

// Configuration
const NUM_SYBIL_NODES = 3;  // Number of Sybil nodes to create
const SYBIL_CONTROLLER_DESCRIPTION = 'SybilMasterCoordinator'; // For logging/description

// Configuration for Bribery Attack (Scenario D)
const NUM_BRIBED_NODES = 2; // Number of nodes to attempt to bribe
const BRIBER_IDENTIFIER = 'MaliciousBriberScenarioD'; // Identifier for the bribing entity in Scenario D

// Helper function for delays
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

// --- Transaction Queue System ---
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
                // For actions that don't return a typical tx response
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
            });
        }
    });
}
// --- End Transaction Queue System ---

// Contract Enums (mirroring NodeManagement.sol)
const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4, Unassigned: 5 };
const ContractNodeType = { Primary: 0, Secondary: 1, Unspecified: 2 };
const BatchVoteType = { NotVoted: 0, Approve: 1, Deny: 2 };

// --- Bribery Attack Simulation ---
async function simulateBriberyAttack(bribingEntitySigner, context, sybilAttackLog, signers, addToTransactionQueue, ethersInstance) {
    console.log(`  --- Starting Bribery Attack Simulation (Scenario D) ---`);
    
    // Initialize scenarioD details if not already present
    if (!sybilAttackLog.scenarioD.details) {
        sybilAttackLog.scenarioD.details = {};
    }
    
    sybilAttackLog.scenarioD.details.briber = bribingEntitySigner.address;
    sybilAttackLog.scenarioD.details.briberIdentifier = BRIBER_IDENTIFIER;
    sybilAttackLog.scenarioD.details.timestamp = DateTime.now().toISO();

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
        // Using bribe calculation logic
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
                    timestamp: DateTime.now().toISO(),
                };
                if (error) {
                    console.error(`      [BRIBERY_CALLBACK] Failed - ${bribeActionDescription}: ${error.message}`);
                    bribeLogEntry.status = 'failed';
                    bribeLogEntry.error = error.message;
                } else {
                    console.log(`      [BRIBERY_CALLBACK] Success - ${bribeActionDescription}. Tx: ${receipt.hash}`);
                    bribeLogEntry.status = 'success';
                    bribeLogEntry.txHash = receipt.hash;
                    
                    // Ensure bribedNodes array exists
                    if (!sybilAttackLog.scenarioD.bribedNodes) {
                        sybilAttackLog.scenarioD.bribedNodes = [];
                    }
                    
                    sybilAttackLog.scenarioD.bribedNodes.push({
                        role: node.role,
                        address: node.address,
                        bribeAmountETH: ethersInstance.utils.formatEther(bribeAmount),
                        txHash: receipt.hash,
                        timestamp: bribeLogEntry.timestamp,
                        expectedBehaviorChanges: getBehaviorChangesForRole(node.role)
                    });
                    successfulBribes++;
                }
                
                // Ensure actions array exists
                if (!sybilAttackLog.scenarioD.actions) {
                    sybilAttackLog.scenarioD.actions = [];
                }
                
                sybilAttackLog.scenarioD.actions.push(bribeLogEntry);
            }
        );
    }

    console.log("    Queueing bribe payments. Simulated behavioral changes will be logged based on successful bribes.");
    sybilAttackLog.scenarioD.outcome = `Attempting to bribe ${nodesToBribe.length} nodes. Status of bribes will be updated after transaction processing.`;
    console.log(`  --- Bribery Attack Simulation Logic Queued ---`);
}

function getBehaviorChangesForRole(role) {
    // Define expected behavior changes based on role
    const behaviorChanges = {
        'Arbitrator': [
            'Biased dispute resolutions favoring the briber',
            'Delayed responses to non-briber disputes',
            'Increased approval rate for briber-related transactions'
        ],
        'Transporter1': [
            'Approval of suspicious batches',
            'Collusion in batch validation',
            'Prioritization of briber-related shipments'
        ],
        'Transporter2': [
            'Approval of suspicious batches',
            'Collusion in batch validation',
            'Prioritization of briber-related shipments'
        ],
        'Transporter3': [
            'Approval of suspicious batches',
            'Collusion in batch validation',
            'Prioritization of briber-related shipments'
        ],
        'Retailer': [
            'False/inflated endorsements for briber products',
            'Preferential treatment in marketplace listings',
            'Manipulation of product ratings'
        ],
        'Manufacturer': [
            'Quality control bypass for briber',
            'Falsification of product specifications',
            'Prioritization of briber orders'
        ],
        'Customer': [
            'False positive reviews',
            'Suppression of negative feedback',
            'Artificial demand creation'
        ]
    };
    
    return behaviorChanges[role] || ['Generic malicious behavior'];
}

function logSimulatedBehavioralChanges(sybilAttackLog, bribingEntitySignerAddress) {
    console.log("    Logging simulated behavioral changes of successfully bribed nodes...");
    const bribedNodesInLog = sybilAttackLog.scenarioD.bribedNodes || [];

    if (bribedNodesInLog.length === 0) {
        console.log("      No nodes were successfully bribed, so no behavioral changes to log.");
        return;
    }

    // Ensure simulatedBehavioralChanges array exists
    if (!sybilAttackLog.scenarioD.simulatedBehavioralChanges) {
        sybilAttackLog.scenarioD.simulatedBehavioralChanges = [];
    }

    for (const bribedNode of bribedNodesInLog) {
        const behaviorChanges = bribedNode.expectedBehaviorChanges || getBehaviorChangesForRole(bribedNode.role);
        
        const simBehaviorLog = {
            type: "simulatedBehaviorChange",
            actorRole: bribedNode.role,
            actorAddress: bribedNode.address,
            description: `${bribedNode.role} ${bribedNode.address} is considered bribed. Expected behaviors: ${behaviorChanges.join(', ')}`,
            expectedBehaviors: behaviorChanges,
            bribedBy: bribingEntitySignerAddress,
            bribeAmount: bribedNode.bribeAmountETH,
            timestamp: DateTime.now().toISO()
        };
        
        sybilAttackLog.scenarioD.simulatedBehavioralChanges.push(simBehaviorLog);
        console.log(`      - ${simBehaviorLog.description}`);
    }
}

async function main() {
    // --- Initial Setup ---
    console.log("=== INITIALIZING SIMULATION ENVIRONMENT ==="); 
    const signers = await ethers.getSigners();
    if (signers.length < 11) { // Base requirement for deployer + 3 sybils + other roles from context. Briber will use one of the 'other' available signers.
        throw new Error(`Need at least 11 signers. Found ${signers.length}. Deployer, ${NUM_SYBIL_NODES} Sybils, 1 Briber, and other roles defined in demo_context.json are required.`);
    }
    const deployerSigner = signers[0]; 
    const sybilSigners = signers.slice(8, 8 + NUM_SYBIL_NODES); 
    const briberSigner = signers[7]; // Assigning a specific signer for the briber

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

    // Initialize log object with improved structure
    const sybilAttackLog = {
        simulationDate: DateTime.now().toISO(),
        simulationStartTime: DateTime.now().toISO(),
        contractAddress: supplyChainNFTAddress,
        sybilController: SYBIL_CONTROLLER_DESCRIPTION,
        numSybilNodes: NUM_SYBIL_NODES,
        sybilNodes: [],
        scenarioA: { 
            description: "Sybil nodes attempt to gain high reputation rapidly through self-referrals or controlled interactions (simulated by admin setting high initial reputation).", 
            actions: [], 
            outcome: "",
            details: {}
        },
        scenarioB: { 
            description: "Sybil nodes collude to dominate validation/arbitration processes (simulated by coordinated voting on a pre-existing batch).", 
            actions: [], 
            outcome: "",
            details: {}
        },
        scenarioC: { 
            description: "Sybil nodes propose and approve a counterfeit/invalid batch of transactions.", 
            actions: [], 
            outcome: "",
            details: {}
        },
        scenarioD: {
            description: "A malicious entity (briber) attempts to influence network participants by sending them direct payments (bribes). Bribed nodes are expected to alter their behavior.",
            briberAddress: briberSigner.address,
            briberIdentifier: BRIBER_IDENTIFIER,
            numNodesTargetedForBribe: NUM_BRIBED_NODES,
            bribedNodes: [],   // Stores info about successfully bribed nodes
            actions: [],       // Log of bribe transaction attempts
            simulatedBehavioralChanges: [], // Log of expected changes from bribed nodes
            outcome: "Bribery attempts logged. Behavioral changes are simulated for FL model training.",
            details: {}
        },
        // Add a new section for FL model integration metadata
        flIntegrationMetadata: {
            sybilNodeAddresses: [], // Will be populated with addresses
            bribedNodeAddresses: [], // Will be populated with addresses
            attackTimestamps: {
                scenarioA: null,
                scenarioB: null,
                scenarioC: null,
                scenarioD: null
            },
            featureSignals: {
                // Will contain signals that FL models should look for
                sybilDetection: [],
                batchMonitoring: [],
                nodeBehavior: [],
                disputeRisk: []
            }
        }
    };

    console.log("\n=== SIMULATING SYBIL ATTACK ===");

    // Ensure context is loaded
    if (!fs.existsSync(contextFilePath)) {
        console.error(`Error: demo_context.json not found at ${contextFilePath}`);
        process.exit(1);
    }

    // --- Register Sybil Nodes ---
    console.log("\n--- Registering Sybil Nodes ---");
    for (let i = 0; i < sybilSigners.length; i++) {
        const sybilNodeSigner = sybilSigners[i];
        const sybilNodeAddress = sybilNodeSigner.address;
        console.log(`  Registering Sybil Node ${i+1}: ${sybilNodeAddress}`);

        // Create Sybil node entry in log
        const sybilNodeEntry = {
            id: `Sybil_${i+1}`,
            address: sybilNodeAddress,
            initialization: [],
            activities: []
        };
        sybilAttackLog.sybilNodes.push(sybilNodeEntry);
        sybilAttackLog.flIntegrationMetadata.sybilNodeAddresses.push(sybilNodeAddress);

        // Set node as verified
        await addToTransactionQueue(
            `Set Sybil ${i+1} (${sybilNodeAddress}) as verified node`,
            () => supplyChainNFT.connect(deployerSigner).setVerifiedNode(sybilNodeAddress, true, gasOptions),
            (error, receipt) => {
                const actionEntry = {
                    action: "setVerifiedNode",
                    verified: true,
                    actor: deployerSigner.address,
                    target: sybilNodeAddress,
                    timestamp: DateTime.now().toISO()
                };
                if (error) {
                    console.error(`    Failed to set Sybil ${i+1} as verified: ${error.message}`);
                    actionEntry.status = "failed";
                    actionEntry.error = error.message;
                } else {
                    console.log(`    Sybil ${i+1} set as verified. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                    actionEntry.status = "success";
                    actionEntry.txHash = receipt ? receipt.hash : null;
                }
                sybilNodeEntry.initialization.push(actionEntry);
            }
        );

        // Set node role (Secondary)
        await addToTransactionQueue(
            `Set Sybil ${i+1} (${sybilNodeAddress}) role to Secondary`,
            () => supplyChainNFT.connect(deployerSigner).setRole(sybilNodeAddress, ContractRole.Transporter, gasOptions),
            (error, receipt) => {
                const actionEntry = {
                    action: "setRole",
                    role: "Transporter",
                    roleValue: ContractRole.Transporter,
                    actor: deployerSigner.address,
                    target: sybilNodeAddress,
                    timestamp: DateTime.now().toISO()
                };
                if (error) {
                    console.error(`    Failed to set Sybil ${i+1} role: ${error.message}`);
                    actionEntry.status = "failed";
                    actionEntry.error = error.message;
                } else {
                    console.log(`    Sybil ${i+1} role set to Transporter. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                    actionEntry.status = "success";
                    actionEntry.txHash = receipt ? receipt.hash : null;
                }
                sybilNodeEntry.initialization.push(actionEntry);
            }
        );

        // Set node type (Secondary)
        await addToTransactionQueue(
            `Set Sybil ${i+1} (${sybilNodeAddress}) node type to Secondary`,
            () => supplyChainNFT.connect(deployerSigner).setNodeType(sybilNodeAddress, ContractNodeType.Secondary, gasOptions),
            (error, receipt) => {
                const actionEntry = {
                    action: "setNodeType",
                    nodeType: "Secondary",
                    nodeTypeValue: ContractNodeType.Secondary,
                    actor: deployerSigner.address,
                    target: sybilNodeAddress,
                    timestamp: DateTime.now().toISO()
                };
                if (error) {
                    console.error(`    Failed to set Sybil ${i+1} node type: ${error.message}`);
                    actionEntry.status = "failed";
                    actionEntry.error = error.message;
                } else {
                    console.log(`    Sybil ${i+1} node type set to Secondary. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                    actionEntry.status = "success";
                    actionEntry.txHash = receipt ? receipt.hash : null;
                }
                sybilNodeEntry.initialization.push(actionEntry);
            }
        );

        // Set initial reputation (artificially high for Sybil detection)
        const initialReputation = 50 + Math.floor(Math.random() * 30); // 50-79
        await addToTransactionQueue(
            `Set Sybil ${i+1} (${sybilNodeAddress}) initial reputation to ${initialReputation}`,
            () => supplyChainNFT.connect(deployerSigner).adminUpdateReputation(sybilNodeAddress, initialReputation, gasOptions),
            (error, receipt) => {
                const actionEntry = {
                    action: "adminUpdateReputation",
                    newReputation: initialReputation,
                    actor: deployerSigner.address,
                    target: sybilNodeAddress,
                    timestamp: DateTime.now().toISO()
                };
                if (error) {
                    console.error(`    Failed to set Sybil ${i+1} reputation: ${error.message}`);
                    actionEntry.status = "failed";
                    actionEntry.error = error.message;
                } else {
                    console.log(`    Sybil ${i+1} reputation set to ${initialReputation}. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                    actionEntry.status = "success";
                    actionEntry.txHash = receipt ? receipt.hash : null;
                }
                sybilNodeEntry.initialization.push(actionEntry);
            }
        );
    }

    console.log("  Processing Sybil node registration in queue...");
    await processTransactionQueue();
    console.log("--- Sybil Node Registration Complete ---");
    sybilAttackLog.flIntegrationMetadata.attackTimestamps.registration = DateTime.now().toISO();

    // --- Scenario A: Sybil nodes file frivolous disputes ---
    console.log("\n--- Scenario A: Sybil Nodes File Frivolous Disputes ---");
    sybilAttackLog.scenarioA.details.startTime = DateTime.now().toISO();
    
    // Find a target token ID from context
    const targetTokenId = context.productDetails && context.productDetails.length > 0 ? 
                          context.productDetails[0].tokenId : null;
    
    if (targetTokenId) {
        console.log(`  Target Token ID for frivolous disputes: ${targetTokenId}`);
        sybilAttackLog.scenarioA.details.targetTokenId = targetTokenId;
        
        // Each Sybil node files a frivolous dispute
        for (let i = 0; i < sybilSigners.length; i++) {
            const sybilNodeSigner = sybilSigners[i];
            const sybilNodeAddress = sybilNodeSigner.address;
            const disputeReason = `Frivolous dispute ${i+1} from Sybil node ${sybilNodeAddress}`;
            const evidenceData = JSON.stringify({
                timestamp: DateTime.now().toISO(),
                disputeReason: disputeReason,
                fabricatedEvidence: true
            });
            
            await addToTransactionQueue(
                `Sybil ${i+1} (${sybilNodeAddress}) files frivolous dispute for Token ID ${targetTokenId}`,
                () => supplyChainNFT.connect(sybilNodeSigner).openDispute(targetTokenId, disputeReason, evidenceData, gasOptions),
                (error, receipt) => {
                    const actionEntry = {
                        type: 'frivolousDispute',
                        tokenId: targetTokenId,
                        reason: disputeReason,
                        sybilNode: sybilNodeAddress,
                        timestamp: DateTime.now().toISO()
                    };
                    if (error) {
                        console.error(`    Failed - Sybil ${i+1} (${sybilNodeAddress}) frivolous dispute: ${error.message}`);
                        actionEntry.status = 'failed';
                        actionEntry.error = error.message;
                    } else {
                        console.log(`    Success - Sybil ${i+1} (${sybilNodeAddress}) filed frivolous dispute. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                        actionEntry.status = 'success';
                        actionEntry.txHash = receipt ? receipt.hash : null;
                        
                        // Extract dispute ID from event if available
                        if (receipt && receipt.events) {
                            const disputeOpenedEvent = receipt.events.find(e => e.event === 'DisputeOpened');
                            if (disputeOpenedEvent && disputeOpenedEvent.args) {
                                actionEntry.disputeId = disputeOpenedEvent.args.disputeId.toString();
                            }
                        }
                    }
                    sybilAttackLog.scenarioA.actions.push(actionEntry);
                    sybilAttackLog.sybilNodes[i].activities.push(actionEntry);
                }
            );
        }
    } else {
        console.warn("  Warning: No target token ID found in context. Skipping frivolous disputes (Scenario A).");
        sybilAttackLog.scenarioA.outcome = "Skipped: No target token ID found in context.";
    }
    
    console.log("  Processing frivolous disputes in queue...");
    await processTransactionQueue();
    console.log("--- Scenario A: Frivolous Disputes Complete ---");
    sybilAttackLog.flIntegrationMetadata.attackTimestamps.scenarioA = DateTime.now().toISO();

    // --- Scenario B: Sybil nodes collude on batch validation ---
    console.log("\n--- Scenario B: Sybil Nodes Collude on Batch Validation ---");
    sybilAttackLog.scenarioB.details.startTime = DateTime.now().toISO();
    
    // Find a target batch ID from context
    const targetBatchId = context.targetBatchIdForSybilTest || 
                         (context.batchDetails && context.batchDetails.length > 0 ? 
                          context.batchDetails[0].batchId : null);
    
    if (targetBatchId) {
        console.log(`  Target Batch ID for collusive voting: ${targetBatchId}`);
        sybilAttackLog.scenarioB.details.targetBatchId = targetBatchId;
        
        // All Sybil nodes vote the same way (Deny) on the batch
        for (let i = 0; i < sybilSigners.length; i++) {
            const sybilNodeSigner = sybilSigners[i];
            const sybilNodeAddress = sybilNodeSigner.address;
            const voteDecision = BatchVoteType.Deny; // All Sybils vote to Deny

            await addToTransactionQueue(
                `Sybil ${i+1} (${sybilNodeAddress}): Vote ${Object.keys(BatchVoteType).find(k => BatchVoteType[k] === voteDecision)} on Batch ${targetBatchId}`,
                () => supplyChainNFT.connect(sybilNodeSigner).castBatchVote(targetBatchId, voteDecision, gasOptions), 
                (error, receipt) => { 
                    const actionEntry = {
                        type: 'collusiveVoting',
                        batchId: targetBatchId,
                        vote: Object.keys(BatchVoteType).find(k => BatchVoteType[k] === voteDecision),
                        sybilNode: sybilNodeAddress,
                        timestamp: DateTime.now().toISO()
                    };
                    if (error) {
                        console.error(`    Failed - Sybil ${i+1} (${sybilNodeAddress}) castBatchVote on batch ${targetBatchId}: ${error.message}`);
                        actionEntry.status = 'failed';
                        actionEntry.error = error.message;
                    } else {
                        console.log(`    Success - Sybil ${i+1} (${sybilNodeAddress}) castBatchVote on batch ${targetBatchId} as ${actionEntry.vote}. Tx: ${receipt ? receipt.hash : 'N/A'}`);
                        actionEntry.status = 'success';
                        actionEntry.txHash = receipt ? receipt.hash : null;
                    }
                    sybilAttackLog.sybilNodes[i].activities.push(actionEntry);
                    sybilAttackLog.scenarioB.actions.push(actionEntry);
                }
            );
        }
    } else {
        sybilAttackLog.scenarioB.outcome = "Skipped: targetBatchIdForSybilTest not found in demo_context.json.";
        console.warn("  Warning: targetBatchIdForSybilTest not found in demo_context.json. Sybil collusion (Scenario B) will be skipped.");
    }
    
    console.log("  Processing Sybil collusive voting in queue...");
    await processTransactionQueue();
    console.log("--- Scenario B: Collusive Voting Complete ---");
    sybilAttackLog.flIntegrationMetadata.attackTimestamps.scenarioB = DateTime.now().toISO();

    // --- Scenario C: Sybil nodes propose and approve counterfeit batch ---
    console.log('\n--- Scenario C: Sybil Nodes Propose and Approve Counterfeit Batch ---');
    sybilAttackLog.scenarioC.details.startTime = DateTime.now().toISO();
    
    const sybilProposer = sybilSigners[0]; // First Sybil will propose

    // Create counterfeit transaction data
    const counterfeitTransactions = [
        {
            from: sybilProposer.address,
            to: ethers.Wallet.createRandom().address,
            tokenId: 999999,
        },
        {
            from: sybilProposer.address,
            to: ethers.Wallet.createRandom().address,
            tokenId: 999998, 
        }
    ];
    
    sybilAttackLog.scenarioC.details.counterfeitTransactions = counterfeitTransactions.map(tx => ({
        from: tx.from,
        to: tx.to,
        tokenId: tx.tokenId.toString()
    }));

    let proposedBatchIdScenarioC;

    // Sybil node 0 proposes counterfeit batch
    await addToTransactionQueue(
        `Sybil Node 0 (${sybilProposer.address}) proposes a counterfeit batch`,
        async () => {
            const tx = await supplyChainNFT.connect(sybilProposer).proposeBatch(counterfeitTransactions, gasOptions);
            const receipt = await tx.wait();
            // Find BatchProposed event to get batchId
            const event = receipt.events?.find(e => e.event === 'BatchProposed');
            if (event && event.args) {
                proposedBatchIdScenarioC = event.args.batchId;
                console.log(`  Counterfeit Batch Proposed with ID: ${proposedBatchIdScenarioC}`);
                sybilAttackLog.scenarioC.actions.push({
                    type: 'counterfeitBatchProposal',
                    action: 'ProposeCounterfeitBatch',
                    batchId: proposedBatchIdScenarioC.toString(),
                    proposerNode: sybilProposer.address,
                    transactions: counterfeitTransactions.map(t => ({...t, tokenId: t.tokenId.toString()})),
                    timestamp: DateTime.now().toISO(),
                    status: 'success',
                    txHash: receipt.hash
                });
                sybilAttackLog.scenarioC.details.counterfeitBatchId = proposedBatchIdScenarioC.toString();
            } else {
                throw new Error("BatchProposed event not found or batchId missing.");
            }
            return tx;
        },
        (error) => {
            if (error) {
                sybilAttackLog.scenarioC.actions.push({
                    type: 'counterfeitBatchProposal',
                    action: 'ProposeCounterfeitBatch',
                    proposerNode: sybilProposer.address,
                    transactions: counterfeitTransactions.map(t => ({...t, tokenId: t.tokenId.toString()})),
                    timestamp: DateTime.now().toISO(),
                    status: 'failed',
                    error: error.message
                });
            }
        }
    );
    await processTransactionQueue(); // Process proposal before voting

    if (proposedBatchIdScenarioC === undefined) {
        console.error('  [SCENARIO C] Failed to propose counterfeit batch. Skipping votes.');
        sybilAttackLog.scenarioC.outcome = 'Failed to propose batch';
    } else {
        // All Sybil nodes vote to approve the counterfeit batch
        for (let i = 0; i < sybilSigners.length; i++) {
            const currentSybil = sybilSigners[i];
            await addToTransactionQueue(
                `Sybil Node ${i} (${currentSybil.address}) votes to APPROVE counterfeit batch ${proposedBatchIdScenarioC}`,
                () => supplyChainNFT.connect(currentSybil).validateBatch(proposedBatchIdScenarioC, true, gasOptions), // true for Approve
                (error, receipt) => {
                    const actionEntry = {
                        type: 'counterfeitBatchApproval',
                        action: 'VoteCounterfeitBatch',
                        batchId: proposedBatchIdScenarioC.toString(),
                        vote: 'Approve',
                        sybilNode: currentSybil.address,
                        timestamp: DateTime.now().toISO()
                    };
                    if (error) {
                        actionEntry.status = 'failed';
                        actionEntry.error = error.message;
                    } else {
                        actionEntry.status = 'success';
                        actionEntry.txHash = receipt ? receipt.hash : null;
                    }
                    sybilAttackLog.scenarioC.actions.push(actionEntry);
                    sybilAttackLog.sybilNodes[i].activities.push(actionEntry);
                }
            );
        }
        await processTransactionQueue(); // Process all votes

        // Check batch status after voting
        try {
            const batchDetails = await supplyChainNFT.getBatchDetails(proposedBatchIdScenarioC);
            console.log(`  [SCENARIO C] Details of counterfeit batch ${proposedBatchIdScenarioC}:`, {
                validated: batchDetails.validated,
                committed: batchDetails.committed,
                approvals: batchDetails.approvals ? batchDetails.approvals.toString() : (await supplyChainNFT.batchApprovals(proposedBatchIdScenarioC)).toString(),
                denials: batchDetails.denials ? batchDetails.denials.toString() : (await supplyChainNFT.batchDenials(proposedBatchIdScenarioC)).toString(),
                proposer: batchDetails.proposer,
                flagged: batchDetails.flagged
            });
            sybilAttackLog.scenarioC.outcome = {
                message: 'Sybil nodes attempted to approve a counterfeit batch.',
                batchId: proposedBatchIdScenarioC.toString(),
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
    console.log("--- Scenario C: Counterfeit Batch Complete ---");
    sybilAttackLog.flIntegrationMetadata.attackTimestamps.scenarioC = DateTime.now().toISO();

    // --- Scenario D: Bribery Attack ---
    console.log('\n--- Scenario D: Simulating Bribery Attack ---');
    sybilAttackLog.scenarioD.details.startTime = DateTime.now().toISO();
    
    await simulateBriberyAttack(briberSigner, context, sybilAttackLog, signers, addToTransactionQueue, ethers);
    
    console.log("  Processing bribery transactions in queue...");
    await processTransactionQueue(); // Process any queued bribery transactions
    
    // Log simulated behavioral changes based on successful bribes
    logSimulatedBehavioralChanges(sybilAttackLog, briberSigner.address);
    
    // Update FL metadata with bribed node addresses
    if (sybilAttackLog.scenarioD.bribedNodes) {
        sybilAttackLog.flIntegrationMetadata.bribedNodeAddresses = 
            sybilAttackLog.scenarioD.bribedNodes.map(node => node.address);
    }
    
    console.log("--- Bribery Attack Simulation (Scenario D) Complete ---");
    sybilAttackLog.flIntegrationMetadata.attackTimestamps.scenarioD = DateTime.now().toISO();

    // Add feature signals for FL models
    sybilAttackLog.flIntegrationMetadata.featureSignals = {
        sybilDetection: [
            "Recently registered nodes with artificially high reputation",
            "Nodes with minimal transaction history but high verification status",
            "Coordinated voting patterns across multiple nodes"
        ],
        batchMonitoring: [
            "Batch proposals with non-existent token IDs",
            "Coordinated approval of suspicious batches",
            "Consistent denial of legitimate batches"
        ],
        nodeBehavior: [
            "Sudden changes in voting patterns after receiving bribes",
            "Abnormal transaction frequency or value spikes",
            "Coordinated actions across multiple nodes in short timeframes"
        ],
        disputeRisk: [
            "Multiple frivolous disputes filed in short succession",
            "Disputes filed by nodes with minimal transaction history",
            "Coordinated dispute filing patterns"
        ]
    };

    // Finalize log
    sybilAttackLog.simulationEndTime = DateTime.now().toISO();
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
