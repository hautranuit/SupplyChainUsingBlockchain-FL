// Simplified Sybil Attack Simulation - 6 Phases
// Uses only 3 Sybil nodes (signer[8], [9], [10])

const { ethers } = require('hardhat');
const fs = require('fs');
const path = require('path');
const { DateTime } = require('luxon');

// === SIMPLIFIED CONFIGURATION ===
const SYBIL_CONFIG = {
    numNodes: 3,                    // 3 Sybil nodes: signer[8], [9], [10]
    initialReputation: 25,          // Starting reputation for Sybil nodes
    targetReputation: 80,           // Target reputation for Primary Node upgrade
    fakeProductTokenId: "4"         // Token ID for fake product (cloning token ID "2")
};

// Attack Identifiers
const ATTACK_IDENTIFIERS = {
    sybilController: 'SybilCoordinator_v1',
    attackCampaign: `ATTACK_${DateTime.now().toFormat('yyyyMMdd_HHmmss')}`
};

// === UTILITY FUNCTIONS ===
const contextFilePath = path.join(__dirname, 'demo_context.json');

// Helper function for delays
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Read and update context file
function readAndUpdateContext(updateFn) {
    let contextData = {};
    if (fs.existsSync(contextFilePath)) {
        try {
            const fileContent = fs.readFileSync(contextFilePath, 'utf8');
            contextData = JSON.parse(fileContent);
        } catch (error) {
            console.error("Error reading context file:", error);
        }
    }
    const updatedContext = updateFn(contextData);
    try {
        fs.writeFileSync(contextFilePath, JSON.stringify(updatedContext, null, 2));
        console.log(`Context updated in ${contextFilePath}`);
    } catch (error) {
        console.error("Error writing context file:", error);
    }
    return updatedContext;
}

// Contract Enums (mirroring NodeManagement.sol)
const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4, Unassigned: 5 };
const ContractNodeType = { Primary: 0, Secondary: 1, Unspecified: 2 };

// Helper functions to get data from context
function getNodeAddressByRole(context, roleName) {
    if (!context.nodes) return null;
    for (const nodeAddr in context.nodes) {
        const node = context.nodes[nodeAddr];
        if (node.name && node.name.toLowerCase().includes(roleName.toLowerCase())) {
            return nodeAddr;
        }
    }
    return null;
}

function getProductByTokenId(context, tokenId) {
    // Search for product information in node interactions
    if (!context.nodes) return null;
    
    // Look for MintProduct interactions to get product details
    for (const nodeAddr in context.nodes) {
        const node = context.nodes[nodeAddr];
        if (node.interactions) {
            for (const interaction of node.interactions) {
                if (interaction.type === "MintProduct" && interaction.tokenId === tokenId) {
                    // Found the minting interaction, create a product object
                    return {
                        tokenId: tokenId,
                        uniqueProductID: `DEMO_PROD_${tokenId.padStart(3, '0')}`,
                        batchNumber: `BATCH_${tokenId}_${Date.now()}`,
                        manufacturingDate: new Date().toISOString().split('T')[0],
                        expirationDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 1 year from now
                        productType: `Product Type ${tokenId}`,
                        manufacturerID: `MFG_${nodeAddr.slice(0, 8)}`,
                        quickAccessURL: `https://demo-product-${tokenId}.example.com`,
                        nftReference: `ipfs://demo-nft-${tokenId}`,
                        currentOwnerAddress: nodeAddr,
                        mintedBy: nodeAddr,
                        mintTimestamp: interaction.timestamp
                    };
                }
            }
        }
    }
    
    // If not found, create a basic product object for token ID "2" (commonly referenced)
    if (tokenId === "2") {
        return {
            tokenId: "2",
            uniqueProductID: "DEMO_PROD_002",
            batchNumber: "BATCH_002_DEMO",
            manufacturingDate: "2025-01-15",
            expirationDate: "2026-01-15",
            productType: "Electronics",
            manufacturerID: "MFG_DEMO_002",
            quickAccessURL: "https://demo-product-2.example.com",
            nftReference: "ipfs://demo-nft-002",
            currentOwnerAddress: "0x04351e7df40d04b5e610c4aa033facf435b98711" // Default to manufacturer
        };
    }
    
    return null;
}

// === NEW/IMPROVED HELPERS ===
// Add an interaction to a node, creating the node/interactions array if needed
function addInteractionToNode(context, address, interaction) {
    if (!context.nodes) context.nodes = {};
    if (!context.nodes[address]) {
        context.nodes[address] = { address, interactions: [] };
    }
    if (!Array.isArray(context.nodes[address].interactions)) {
        context.nodes[address].interactions = [];
    }
    context.nodes[address].interactions.push(interaction);
}

// Mark a node as Sybil with all relevant flags
function markNodeAsSybil(context, sybilNode, campaign, nodeTypeName) {
    if (!context.nodes) context.nodes = {};
    if (!context.nodes[sybilNode.address]) {
        context.nodes[sybilNode.address] = {
            address: sybilNode.address,
            name: sybilNode.name,
            role: sybilNode.role,
            roleName: "Transporter",
            nodeType: sybilNode.nodeType,
            nodeTypeName,
            initialReputation: sybilNode.initialReputation,
            currentReputation: sybilNode.currentReputation,
            isVerified: true,
            isSybil: true,
            attackCampaign: campaign,
            suspiciousActivity: true,
            interactions: []
        };
    } else {
        Object.assign(context.nodes[sybilNode.address], {
            isSybil: true,
            attackCampaign: campaign,
            suspiciousActivity: true,
            currentReputation: sybilNode.currentReputation,
            nodeType: sybilNode.nodeType,
            nodeTypeName
        });
    }
}

// Mark a node as bribed
function markNodeAsBribed(context, address, bribeRecord) {
    if (!context.nodes) context.nodes = {};
    if (!context.nodes[address]) {
        context.nodes[address] = { address, interactions: [] };
    }
    Object.assign(context.nodes[address], {
        receivedBribe: true,
        bribeAmount: bribeRecord.amount,
        briberAddress: bribeRecord.from,
        suspiciousActivity: true
    });
}

// === MAIN EXECUTION ===
async function main() {
    console.log("=== SYBIL ATTACK SIMULATION - 6 PHASES ===");
    console.log(`Campaign: ${ATTACK_IDENTIFIERS.attackCampaign}`);
    console.log(`Controller: ${ATTACK_IDENTIFIERS.sybilController}`);
    
    // Load context and setup
    const context = JSON.parse(fs.readFileSync(contextFilePath, 'utf8'));
    const contractAddress = context.contractAddress;
    if (!contractAddress) {
        throw new Error("Contract address not found in context");
    }

    const signers = await ethers.getSigners();
    if (signers.length < 11) {
        throw new Error(`Need at least 11 signers. Found ${signers.length}`);
    }

    // Setup signers
    const deployerSigner = signers[0];
    const sybilSigners = [signers[8], signers[9], signers[10]]; // 3 Sybil nodes
    const buyerSigner = signers.find(s => s.address.toLowerCase() === getNodeAddressByRole(context, "buyer"));
    const retailerSigner = signers.find(s => s.address.toLowerCase() === getNodeAddressByRole(context, "retailer"));

    console.log("Deployer:", deployerSigner.address);
    console.log("Sybil Nodes:", sybilSigners.map(s => s.address));
    console.log("Buyer:", buyerSigner?.address);
    console.log("Retailer:", retailerSigner?.address);

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployerSigner);
    
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('30', 'gwei'),
        maxFeePerGas: ethers.parseUnits('100', 'gwei')
    };

    // Initialize attack log
    const attackLog = {
        timestamp: DateTime.now().toISO(),
        campaign: ATTACK_IDENTIFIERS.attackCampaign,
        phases: [],
        sybilNodes: [],
        fakeProducts: [],
        bribery: [],
        flTrainingData: {
            sybilDetection: [],
            briberyDetection: []
        }
    };

    // ==================== PHASE 1: ĐĂNG KÝ CÁC NODE SYBIL ====================
    console.log("\n🎭 PHASE 1: Register Sybil nodes");
    
    const phase1Log = {
        phase: 1,
        description: "Đăng ký các node Sybil",
        startTime: DateTime.now().toISO(),
        actions: []
    };

    for (let i = 0; i < SYBIL_CONFIG.numNodes; i++) {
        const signer = sybilSigners[i];
        const sybilName = `Sybil_Node_${i + 1}`;
        
        console.log(`  Registering ${sybilName} (${signer.address})`);
        
        // Set as verified node
        console.log(`    Setting as verified node...`);
        const verifyTx = await supplyChainNFT.connect(deployerSigner).setVerifiedNode(signer.address, true, gasOptions);
        await verifyTx.wait();
        
        // Set role as Retailer (not Transporter)
        console.log(`    Setting role as Retailer...`);
        const roleTx = await supplyChainNFT.connect(deployerSigner).setRole(signer.address, ContractRole.Retailer, gasOptions);
        await roleTx.wait();
        
        // Set node type as Secondary
        console.log(`    Setting node type as Secondary...`);
        const nodeTypeTx = await supplyChainNFT.connect(deployerSigner).setNodeType(signer.address, ContractNodeType.Secondary, gasOptions);
        await nodeTypeTx.wait();
        
        // Set initial reputation
        console.log(`    Setting initial reputation to ${SYBIL_CONFIG.initialReputation}...`);
        const repTx = await supplyChainNFT.connect(deployerSigner).adminUpdateReputation(signer.address, SYBIL_CONFIG.initialReputation, gasOptions);
        await repTx.wait();
        
        const sybilNode = {
            id: `SYBIL_${i + 1}`,
            name: sybilName,
            address: signer.address,
            role: ContractRole.Retailer, // Changed to Retailer
            nodeType: ContractNodeType.Secondary,
            initialReputation: SYBIL_CONFIG.initialReputation,
            currentReputation: SYBIL_CONFIG.initialReputation
        };
        
        attackLog.sybilNodes.push(sybilNode);
        phase1Log.actions.push({
            action: "registerSybilNode",
            nodeId: sybilNode.id,
            address: signer.address,
            timestamp: DateTime.now().toISO()
        });
        
        // Add to FL training data
        attackLog.flTrainingData.sybilDetection.push({
            nodeAddress: signer.address,
            isSybil: true,
            registrationPattern: "coordinated_registration",
            initialReputation: SYBIL_CONFIG.initialReputation,
            timestamp: DateTime.now().toUnixInteger()
        });
        
        await delay(2000); // 2 second delay between registrations
    }
    
    phase1Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase1Log);
    console.log("✅ PHASE 1 Complete: 3 Sybil nodes registered");

    // === RESET TOKEN OWNERSHIP BEFORE PHASE 2 ===
    // Correct original owners for tokens 1, 2, 3
    const originalOwners = {
        "1": "0x724876f86fA52568aBc51955BD3A68bFc1441097", 
        "2": "0x72EB9742d3B684ebA40F11573b733Ac9dB499f23",
        "3": "0x72EB9742d3B684ebA40F11573b733Ac9dB499f23"
    };
    for (const tokenId of ["1", "2", "3"]) {
        if (!originalOwners[tokenId]) continue;
        let currentOwner;
        try {
            currentOwner = await supplyChainNFT.ownerOf(tokenId);
        } catch (err) {
            continue;
        }
        if (currentOwner.toLowerCase() !== originalOwners[tokenId].toLowerCase()) {
            const currentSigner = signers.find(s => s.address.toLowerCase() === currentOwner.toLowerCase());
            if (currentSigner) {
                try {
                    await supplyChainNFT.connect(currentSigner).approve(originalOwners[tokenId], tokenId, gasOptions);
                } catch (e) {}
                try {
                    await supplyChainNFT.connect(currentSigner).transferFrom(currentOwner, originalOwners[tokenId], tokenId, gasOptions);
                } catch (e) {}
            }
        }
    }

    // ==================== PHASE 2: PRODUCT TRANSACTIONS ====================
    console.log("\n💰 PHASE 2: Product transactions between Buyer/Retailer and Sybil nodes");
    const phase2Log = {
        phase: 2,
        description: "Product transactions and ownership verification",
        startTime: DateTime.now().toISO(),
        actions: []
    };
    const products = ["1", "2", "3"];
    const polAmounts = ["0.2", "0.4", "0.6"];
    for (let i = 0; i < products.length && i < SYBIL_CONFIG.numNodes; i++) {
        const tokenId = products[i];
        const sybilSigner = sybilSigners[i];
        const polAmount = ethers.parseEther(polAmounts[i]);
        // Check current owner before attempting transfer
        let currentOwner;
        try {
            currentOwner = await supplyChainNFT.ownerOf(tokenId);
            console.log(`    [Check] Token ${tokenId} current owner: ${currentOwner}`);
        } catch (err) {
            console.error(`    [Error] Could not fetch owner for token ${tokenId}:`, err.message);
            phase2Log.actions.push({
                action: "checkOwnerFailed",
                tokenId,
                error: err.message,
                timestamp: DateTime.now().toISO()
            });
            continue;
        }
        // Dùng signer là owner hiện tại để chuyển token cho sybil node
        const currentSigner = signers.find(s => s.address.toLowerCase() === currentOwner.toLowerCase());
        if (!currentSigner) {
            console.warn(`    [Skip] No signer for current owner ${currentOwner} of token ${tokenId}`);
            phase2Log.actions.push({
                action: "noSignerForOwner",
                tokenId,
                owner: currentOwner,
                to: sybilSigner.address,
                timestamp: DateTime.now().toISO()
            });
            continue;
        }
        if (currentOwner.toLowerCase() === sybilSigner.address.toLowerCase()) {
            console.log(`    [Skip] Token ${tokenId} already owned by Sybil node ${sybilSigner.address}`);
            phase2Log.actions.push({
                action: "alreadySybilOwner",
                tokenId,
                owner: currentOwner,
                to: sybilSigner.address,
                timestamp: DateTime.now().toISO()
            });
            continue;
        }
        try {
            // Approve nếu cần
            try {
                await supplyChainNFT.connect(currentSigner).approve(sybilSigner.address, tokenId, gasOptions);
                console.log(`    [Approve] Owner ${currentOwner} approved Sybil node ${sybilSigner.address} for token ${tokenId}`);
            } catch (e) {
                console.log(`    [Approve] (Optional) Approve failed or unnecessary for token ${tokenId}: ${e.message}`);
            }
            const transferTx = await supplyChainNFT.connect(currentSigner).transferFrom(currentOwner, sybilSigner.address, tokenId, gasOptions);
            console.log(`    [Transfer] Token ${tokenId} transferred from ${currentOwner} to ${sybilSigner.address}, tx: ${transferTx.hash}`);
            // Check new owner
            const newOwner = await supplyChainNFT.ownerOf(tokenId);
            console.log(`    [Check] New owner of token ${tokenId}: ${newOwner}`);
            phase2Log.actions.push({
                action: "transferNFT",
                tokenId,
                from: currentOwner,
                to: sybilSigner.address,
                txHash: transferTx.hash,
                newOwner,
                timestamp: DateTime.now().toISO()
            });
        } catch (err) {
            console.error(`    [Error] Transfer failed for token ${tokenId} from ${currentOwner} to ${sybilSigner.address}:`, err.message);
            phase2Log.actions.push({
                action: "transferFailed",
                tokenId,
                from: currentOwner,
                to: sybilSigner.address,
                error: err.message,
                timestamp: DateTime.now().toISO()
            });
            continue;
        }
        // After transfer, verify and retry if not successful
        let newOwner;
        try {
            newOwner = await supplyChainNFT.ownerOf(tokenId);
        } catch (e) { newOwner = undefined; }
        if (newOwner && newOwner.toLowerCase() !== sybilSigner.address.toLowerCase()) {
            // Retry transfer once if failed
            try {
                await supplyChainNFT.connect(currentSigner).approve(sybilSigner.address, tokenId, gasOptions);
            } catch (e) {}
            try {
                const transferTx2 = await supplyChainNFT.connect(currentSigner).transferFrom(currentOwner, sybilSigner.address, tokenId, gasOptions);
                await transferTx2.wait();
                newOwner = await supplyChainNFT.ownerOf(tokenId);
                phase2Log.actions.push({
                    action: "retryTransferNFT",
                    tokenId,
                    from: currentOwner,
                    to: sybilSigner.address,
                    txHash: transferTx2.hash,
                    newOwner,
                    timestamp: DateTime.now().toISO()
                });
            } catch (err2) {
                phase2Log.actions.push({
                    action: "retryTransferFailed",
                    tokenId,
                    from: currentOwner,
                    to: sybilSigner.address,
                    error: err2.message,
                    timestamp: DateTime.now().toISO()
                });
            }
        }
        await delay(2000);
    }
    phase2Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase2Log);
    console.log("✅ PHASE 2 Complete: Product transactions and ownership verified");

    // ==================== PHASE 3: SYBIL NODES PROPOSE AND COMMIT BATCH ====================
    console.log("\n📦 PHASE 3: Sybil nodes propose, validate, and commit multiple batches");
    const phase3Log = {
        phase: 3,
        description: "Propose, validate, and commit multiple batches with validator selection and interleaved reputation update",
        startTime: DateTime.now().toISO(),
        actions: []
    };
    // Use products array from earlier phase
    let sybilReputations = [attackLog.sybilNodes[0].currentReputation, attackLog.sybilNodes[1].currentReputation, attackLog.sybilNodes[2].currentReputation];
    const batchCount = 10;
    // Increase candidate pool for validator selection
    const candidateValidators = signers.slice(1, 7); // 6 candidates (skip deployer)
    for (let round = 0; round < batchCount; round++) {
        const fromIdx = round % 3;
        const toIdx = (round + 1) % 3;
        const tokenId = products[round % 3];
        const batchTransactions = [
            { from: sybilSigners[fromIdx].address, to: sybilSigners[toIdx].address, tokenId: BigInt(tokenId) }
        ];
        const proposerSigner = sybilSigners[fromIdx];
        // Randomly select 3 validators from candidate pool for this batch
        const shuffled = candidateValidators.slice().sort(() => 0.5 - Math.random());
        const selectedValidators = shuffled.slice(0, 3).map(s => s.address);
        let proposeTx, proposeReceipt, batchProposedEvent, batchId;
        try {
            proposeTx = await supplyChainNFT.connect(proposerSigner).proposeBatch(batchTransactions, gasOptions);
            proposeReceipt = await proposeTx.wait();
            if (proposeReceipt.events && Array.isArray(proposeReceipt.events)) {
                batchProposedEvent = proposeReceipt.events.find(e => e.event === 'BatchProposed');
            }
            if (!batchProposedEvent) {
                try {
                    const iface = new ethers.Interface(supplyChainNFT.interface.fragments || supplyChainNFT.interface);
                    for (const log of proposeReceipt.logs) {
                        try {
                            const parsed = iface.parseLog(log);
                            if (parsed.name === 'BatchProposed') {
                                batchProposedEvent = { args: parsed.args };
                                break;
                            }
                        } catch (e) {}
                    }
                } catch (e) {}
            }
            if (batchProposedEvent && batchProposedEvent.args) {
                batchId = batchProposedEvent.args.batchId?.toString();
            } else {
                console.error(`[Error] Could not find BatchProposed event in proposeReceipt (round ${round + 1}).`);
                continue;
            }
            // Log batch proposal details
            console.log(`\n  [BatchProposed] Round ${round + 1}, Batch ID: ${batchId}`);
            console.log(`    Transactions:`, batchTransactions);
            console.log(`    Selected Validators:`, selectedValidators);
            phase3Log.actions.push({
                action: "proposeBatch",
                round: round + 1,
                batchId,
                proposer: proposerSigner.address,
                transactions: batchTransactions,
                selectedValidators,
                timestamp: DateTime.now().toISO()
            });
            // Validators vote
            let approvals = 0;
            let requiredApprovals = 0;
            const validatorVotes = [];
            try {
                const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
                requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajorityFraction) / 100);
                for (let i = 0; i < selectedValidators.length; i++) {
                    const validatorAddr = selectedValidators[i];
                    const validatorSigner = signers.find(s => s.address.toLowerCase() === validatorAddr.toLowerCase());
                    if (!validatorSigner) continue;
                    const vote = approvals < requiredApprovals;
                    if (vote) approvals++;
                    const voteTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote, gasOptions);
                    const voteReceipt = await voteTx.wait();
                    validatorVotes.push({validator: validatorSigner.address, approve: vote, txHash: voteReceipt.transactionHash});
                    phase3Log.actions.push({
                        action: "validateBatch",
                        round: round + 1,
                        batchId,
                        validator: validatorSigner.address,
                        approve: vote,
                        txHash: voteReceipt.transactionHash,
                        timestamp: DateTime.now().toISO()
                    });
                    console.log(`    Validator ${validatorSigner.address} voted ${vote ? 'APPROVE' : 'DENY'} (tx: ${voteReceipt.transactionHash})`);
                    await delay(500);
                }
            } catch (error) {
                console.error(`    ERROR during validator voting (round ${round + 1}):`, error);
            }
            // Commit batch if approvals reached
            if (approvals >= requiredApprovals) {
                let commitTx, commitReceipt, isCommitted = false, isFlagged = false;
                try {
                    commitTx = await supplyChainNFT.connect(deployerSigner).commitBatch(batchId, gasOptions);
                    commitReceipt = await commitTx.wait();
                    const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
                    isCommitted = batchDetails.committed;
                    isFlagged = batchDetails.flagged;
                } catch (error) {
                    commitReceipt = { transactionHash: "N/A (Commit Error)" };
                }
                phase3Log.actions.push({
                    action: "commitBatch",
                    round: round + 1,
                    batchId,
                    txHash: commitReceipt.transactionHash,
                    timestamp: DateTime.now().toISO()
                });
                // Check ownership and log
                for (const tx of batchTransactions) {
                    const owner = await supplyChainNFT.ownerOf(tx.tokenId);
                    phase3Log.actions.push({
                        action: "checkOwner",
                        round: round + 1,
                        tokenId: tx.tokenId.toString(),
                        owner,
                        timestamp: DateTime.now().toISO()
                    });
                    console.log(`    [Ownership] Token ${tx.tokenId} new owner: ${owner}`);
                }
                // Interleaved: Update reputation for all Sybil nodes by +10 after each commit
                if (isCommitted) {
                    for (let i = 0; i < SYBIL_CONFIG.numNodes; i++) {
                        sybilReputations[i] = (sybilReputations[i] || 0) + 10;
                        const signer = sybilSigners[i];
                        const repTx = await supplyChainNFT.connect(deployerSigner).adminUpdateReputation(signer.address, sybilReputations[i], gasOptions);
                        await repTx.wait();
                        attackLog.sybilNodes[i].currentReputation = sybilReputations[i];
                        phase3Log.actions.push({
                            action: "updateReputation",
                            round: round + 1,
                            nodeAddress: signer.address,
                            newReputation: sybilReputations[i],
                            timestamp: DateTime.now().toISO()
                        });
                        console.log(`    [Reputation] ${signer.address} updated to ${sybilReputations[i]}`);
                        attackLog.flTrainingData.sybilDetection.push({
                            nodeAddress: signer.address,
                            reputationChange: sybilReputations[i],
                            reputationManipulation: true,
                            timestamp: DateTime.now().toUnixInteger()
                        });
                    }
                }
                console.log(`  [BatchResult] Round ${round + 1}: Batch ${batchId} committed. Approvals: ${approvals}/${selectedValidators.length}`);
            } else {
                console.log(`  [BatchResult] Round ${round + 1}: Batch ${batchId} NOT committed. Approvals: ${approvals}/${selectedValidators.length}`);
            }
            // Log summary for this batch
            phase3Log.actions.push({
                action: "batchSummary",
                round: round + 1,
                batchId,
                transactions: batchTransactions,
                selectedValidators,
                validatorVotes,
                sybilReputations: sybilReputations.slice(),
                timestamp: DateTime.now().toISO()
            });
            await delay(1000);
        } catch (err) {
            console.error(`[Error] Batch round ${round + 1} failed:`, err.message);
            continue;
        }
    }
    phase3Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase3Log);
    console.log("✅ PHASE 3 Complete: 10 NFT batch transfers and reputation updates interleaved");

    // ==================== PHASE 5: PROMOTE TO PRIMARY NODE ====================
    console.log("\n🚀 PHASE 5: Promote Sybil nodes to Primary Node");
    const phase5Log = {
        phase: 5,
        description: "Thăng cấp thành Primary Node",
        startTime: DateTime.now().toISO(),
        actions: []
    };
    // Only call setNodeType for promotion, do not forcibly set reputation
    for (let i = 0; i < SYBIL_CONFIG.numNodes; i++) {
        const signer = sybilSigners[i];
        console.log(`  Promoting Sybil Node ${i + 1} to Primary Node (Current Reputation: ${attackLog.sybilNodes[i].currentReputation})`);
        const nodeTypeTx = await supplyChainNFT.connect(deployerSigner).setNodeType(signer.address, ContractNodeType.Primary, gasOptions);
        await nodeTypeTx.wait();
        attackLog.sybilNodes[i].nodeType = ContractNodeType.Primary;
        phase5Log.actions.push({
            action: "upgradeToPrimary",
            nodeAddress: signer.address,
            reputation: attackLog.sybilNodes[i].currentReputation,
            timestamp: DateTime.now().toISO()
        });
        // Add to FL training data
        attackLog.flTrainingData.sybilDetection.push({
            nodeAddress: signer.address,
            nodeTypeChange: "Secondary_to_Primary",
            rapidPromotion: true,
            timestamp: DateTime.now().toUnixInteger()
        });
        await delay(2000);
    }
    phase5Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase5Log);
    console.log("✅ PHASE 5 Complete: All Sybil nodes promoted to Primary");

    // ==================== PHASE 6: ATTACK - FAKE PRODUCT & BRIBERY ====================
    console.log("\n⚔️ PHASE 6: Attack - Inject fake product and bribe validators");
    
    const phase6Log = {
        phase: 6,
        description: "Tấn công với hàng giả và hối lộ",
        startTime: DateTime.now().toISO(),
        actions: []
    };

    // 6.1: Create fake product (clone token ID "2" metadata)
    console.log("  6.1: Create fake product (clone token ID '2')");
    
    const originalProduct = getProductByTokenId(context, "2");
    if (originalProduct) {
        // Mint fake product with same metadata as token ID "2"
        const fakeProductData = {
            recipient: sybilSigners[0].address, // Give to first Sybil node
            uniqueProductID: originalProduct.uniqueProductID + "_FAKE", // Mark as fake
            batchNumber: originalProduct.batchNumber + "_COUNTERFEIT",
            manufacturingDate: originalProduct.manufacturingDate,
            expirationDate: originalProduct.expirationDate,
            productType: originalProduct.productType, // Same product type
            manufacturerID: originalProduct.manufacturerID + "_FAKE",
            quickAccessURL: "http://fake-product.malicious.com",
            nftReference: originalProduct.nftReference // Same NFT reference (suspicious!)
        };
        
        console.log(`    Minting fake product (Token ID: ${SYBIL_CONFIG.fakeProductTokenId})`);
        console.log(`    Cloning metadata from Token ID '2'`);
        console.log(`    Fake Product Type: ${fakeProductData.productType}`);
        
        // Mint the fake product
        const mintTx = await supplyChainNFT.connect(deployerSigner).mintNFT(
            {
                recipient: fakeProductData.recipient,
                uniqueProductID: fakeProductData.uniqueProductID,
                batchNumber: fakeProductData.batchNumber,
                manufacturingDate: fakeProductData.manufacturingDate,
                expirationDate: fakeProductData.expirationDate,
                productType: fakeProductData.productType,
                manufacturerID: fakeProductData.manufacturerID,
                quickAccessURL: fakeProductData.quickAccessURL,
                nftReference: fakeProductData.nftReference
            },
            gasOptions
        );
        const mintReceipt = await mintTx.wait();
        
        const fakeProduct = {
            tokenId: SYBIL_CONFIG.fakeProductTokenId,
            originalTokenId: "2",
            ownerAddress: sybilSigners[0].address,
            isCounterfeit: true,
            clonedMetadata: true,
            ...fakeProductData
        };
        
        attackLog.fakeProducts.push(fakeProduct);
        
        phase6Log.actions.push({
            action: "mintFakeProduct",
            tokenId: SYBIL_CONFIG.fakeProductTokenId,
            clonedFrom: "2",
            ownerAddress: sybilSigners[0].address,
            timestamp: DateTime.now().toISO()
        });
        
        // Add to FL training data
        attackLog.flTrainingData.sybilDetection.push({
            nodeAddress: sybilSigners[0].address,
            productCounterfeiting: true,
            originalTokenId: "2",
            fakeTokenId: SYBIL_CONFIG.fakeProductTokenId,
            timestamp: DateTime.now().toUnixInteger()
        });
    }

    // 6.2: Propose malicious batch with fake product
    console.log("  6.2: Propose batch containing fake product");
    
    const maliciousTransactions = [
        { from: sybilSigners[0].address, to: sybilSigners[1].address, tokenId: SYBIL_CONFIG.fakeProductTokenId },
        { from: sybilSigners[1].address, to: buyerSigner?.address || sybilSigners[2].address, tokenId: SYBIL_CONFIG.fakeProductTokenId }
    ];
    
    const maliciousProposeTx = await supplyChainNFT.connect(sybilSigners[0]).proposeBatch(maliciousTransactions, gasOptions);
    const maliciousReceipt = await maliciousProposeTx.wait();
    
    const maliciousBatchEvent = maliciousReceipt.events?.find(e => e.event === 'BatchProposed');
    const maliciousBatchId = maliciousBatchEvent?.args?.batchId?.toString();
    
    console.log(`    Malicious Batch ID: ${maliciousBatchId} (contains fake product)`);

    // 6.3: Sybil nodes vote approve for malicious batch
    console.log("  6.3: Sybil nodes vote to approve batch containing fake product");
    
    if (maliciousBatchId) {
        for (let i = 0; i < 2; i++) { // 2 Sybil nodes approve
            const validatorSigner = sybilSigners[i];
            console.log(`    Sybil Node ${i + 1} approves malicious batch`);
            
            const approveTx = await supplyChainNFT.connect(validatorSigner).validateBatch(maliciousBatchId, true, gasOptions);
            await approveTx.wait();
            
            await delay(1000);
        }
    }

    // 6.4: Bribery attack on legitimate validators
    console.log("  6.4: Bribery attack on other validators");
    
    const bribeAmount = ethers.parseEther("1.0"); // 1 POL bribe
    const legitimateValidators = [
        buyerSigner?.address,
        retailerSigner?.address
    ].filter(addr => addr); // Remove undefined addresses
    
    for (const validatorAddress of legitimateValidators) {
        if (validatorAddress) {
            console.log(`    Sending bribe of ${ethers.formatEther(bribeAmount)} POL to ${validatorAddress}`);
            
            // Send bribe (ETH transfer)
            const bribeTx = await sybilSigners[0].sendTransaction({
                to: validatorAddress,
                value: bribeAmount,
                ...gasOptions
            });
            await bribeTx.wait();
            
            const bribeRecord = {
                from: sybilSigners[0].address,
                to: validatorAddress,
                amount: ethers.formatEther(bribeAmount),
                purpose: "influence_validation",
                maliciousBatchId: maliciousBatchId,
                timestamp: DateTime.now().toISO()
            };
            
            attackLog.bribery.push(bribeRecord);
            
            // Add to FL training data
            attackLog.flTrainingData.briberyDetection.push({
                briberAddress: sybilSigners[0].address,
                targetAddress: validatorAddress,
                amount: ethers.formatEther(bribeAmount),
                briberyType: "validation_influence",
                relatedBatch: maliciousBatchId,
                timestamp: DateTime.now().toUnixInteger()
            });
            
            console.log("    ⚠️ Validator may change behavior after receiving bribe");
            
            await delay(2000);
        }
    }
    
    phase6Log.actions.push({
        action: "executeAttack",
        maliciousBatchId: maliciousBatchId,
        fakeProductTokenId: SYBIL_CONFIG.fakeProductTokenId,
        briberyTargets: legitimateValidators.length,
        totalBribeAmount: ethers.formatEther(bribeAmount * BigInt(legitimateValidators.length)),
        timestamp: DateTime.now().toISO()
    });
    
    phase6Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase6Log);
    console.log("⚔️ PHASE 6 Complete: Attack finished - Fake product and bribery executed");

    // ==================== FINALIZATION ====================
    console.log("\n📊 SYBIL ATTACK SUMMARY");
    
    attackLog.endTime = DateTime.now().toISO();
    attackLog.summary = {
        totalSybilNodes: SYBIL_CONFIG.numNodes,
        fakeProductsCreated: attackLog.fakeProducts.length,
        briberyAttempts: attackLog.bribery.length,
        totalBribeAmount: attackLog.bribery.reduce((sum, b) => sum + parseFloat(b.amount), 0),
        phasesCompleted: attackLog.phases.length,
        flTrainingRecords: {
            sybilDetection: attackLog.flTrainingData.sybilDetection.length,
            briberyDetection: attackLog.flTrainingData.briberyDetection.length
        }
    };

    console.log("Sybil Nodes created:", attackLog.summary.totalSybilNodes);
    console.log("Fake products created:", attackLog.summary.fakeProductsCreated);
    console.log("Bribery attempts:", attackLog.summary.briberyAttempts);
    console.log("Total bribe amount:", attackLog.summary.totalBribeAmount, "POL");
    console.log("FL Training Records:");
    console.log("  - Sybil Detection:", attackLog.summary.flTrainingRecords.sybilDetection);
    console.log("  - Bribery Detection:", attackLog.summary.flTrainingRecords.briberyDetection);

    // Update demo_context.json with ALL attack data for FL integration
    readAndUpdateContext(currentContext => {
        // Initialize attack data structure
        if (!currentContext.attackData) {
            currentContext.attackData = {};
        }
        // Store complete attack log directly in demo_context.json
        currentContext.attackData.sybilAttack = {
            timestamp: attackLog.timestamp,
            campaign: ATTACK_IDENTIFIERS.attackCampaign,
            controller: ATTACK_IDENTIFIERS.sybilController,
            phases: attackLog.phases,
            summary: attackLog.summary,
            sybilNodes: attackLog.sybilNodes,
            fakeProducts: attackLog.fakeProducts,
            briberyRecords: attackLog.bribery,
            flTrainingData: attackLog.flTrainingData
        };
        // Mark Sybil node addresses in nodes section for detection
        for (const sybilNode of attackLog.sybilNodes) {
            markNodeAsSybil(currentContext, sybilNode, ATTACK_IDENTIFIERS.attackCampaign, sybilNode.nodeType === 0 ? "Primary" : "Secondary");
        }
        // Add fake product interactions to Sybil nodes
        for (const fakeProduct of attackLog.fakeProducts) {
            addInteractionToNode(currentContext, fakeProduct.ownerAddress, {
                type: "MintFakeProduct",
                tokenId: fakeProduct.tokenId,
                originalTokenId: fakeProduct.originalTokenId,
                timestamp: DateTime.now().toUnixInteger(),
                details: `Minted counterfeit product ${fakeProduct.uniqueProductID} (clone of token ${fakeProduct.originalTokenId})`,
                isCounterfeit: true,
                clonedMetadata: true,
                detectionMarkers: {
                    suspiciousManufacturerID: fakeProduct.manufacturerID.includes("_FAKE"),
                    suspiciousBatchNumber: fakeProduct.batchNumber.includes("_COUNTERFEIT"),
                    duplicateNFTReference: true,
                    maliciousURL: fakeProduct.quickAccessURL.includes("malicious")
                }
            });
        }
        // Add bribery detection markers to affected nodes
        for (const bribeRecord of attackLog.bribery) {
            markNodeAsBribed(currentContext, bribeRecord.to, bribeRecord);
            addInteractionToNode(currentContext, bribeRecord.to, {
                type: "ReceiveBribe",
                from: bribeRecord.from,
                amount: bribeRecord.amount,
                purpose: bribeRecord.purpose,
                relatedBatch: bribeRecord.maliciousBatchId,
                timestamp: DateTime.now().toUnixInteger(),
                details: `Received bribe of ${bribeRecord.amount} POL from ${bribeRecord.from}`
            });
        }
        // Add attack-related interactions to Sybil nodes
        for (let i = 0; i < attackLog.sybilNodes.length; i++) {
            const sybilNode = attackLog.sybilNodes[i];
            const sybilAddress = sybilNode.address;
            addInteractionToNode(currentContext, sybilAddress, {
                type: "SybilRegistration",
                campaign: ATTACK_IDENTIFIERS.attackCampaign,
                phase: "Phase 1 - Registration",
                timestamp: DateTime.now().toUnixInteger(),
                details: `Registered as Sybil node in attack campaign ${ATTACK_IDENTIFIERS.attackCampaign}`
            });
            addInteractionToNode(currentContext, sybilAddress, {
                type: "ReputationManipulation",
                oldReputation: sybilNode.initialReputation,
                newReputation: sybilNode.currentReputation,
                phase: "Phase 4 - Reputation Update",
                timestamp: DateTime.now().toUnixInteger(),
                details: `Artificially boosted reputation from ${sybilNode.initialReputation} to ${sybilNode.currentReputation}`
            });
            if (sybilNode.nodeType === 0) { // Primary
                addInteractionToNode(currentContext, sybilAddress, {
                    type: "SuspiciousPromotion",
                    oldNodeType: "Secondary",
                    newNodeType: "Primary",
                    phase: "Phase 5 - Primary Upgrade",
                    timestamp: DateTime.now().toUnixInteger(),
                    details: "Rapidly promoted to Primary node through reputation manipulation"
                });
            }
        }
        return currentContext;
    });

    console.log("\n🎯 SYBIL ATTACK SIMULATION COMPLETE!");
    console.log("🔬 Data is ready for FL training with 2 core models:");
    console.log("   - sybil_detection");
    console.log("   - bribery_detection");
    console.log("\n📁 Attack data integrated into:");
    console.log(`   - ${contextFilePath} (complete attack data for FL training)`);
}

// Execute main function
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });
