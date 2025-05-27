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

    // Sau khi hoàn thành phase 1, cập nhật demo_context.json
    readAndUpdateContext(currentContext => {
        if (!currentContext.attackData) currentContext.attackData = {};
        currentContext.attackData.sybilAttack = {
            timestamp: attackLog.timestamp,
            campaign: ATTACK_IDENTIFIERS.attackCampaign,
            controller: ATTACK_IDENTIFIERS.sybilController,
            phases: attackLog.phases,
            sybilNodes: attackLog.sybilNodes,
            fakeProducts: attackLog.fakeProducts,
            briberyRecords: attackLog.bribery,
            flTrainingData: attackLog.flTrainingData
        };
        return currentContext;
    });

    // === RESET TOKEN OWNERSHIP BEFORE PHASE 2 ===
    // Correct original owners for tokens 1, 2, 3
    const originalOwners = {
        "1": "0x04351e7dF40d04B5E610c4aA033faCf435b98711", 
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
    let allSybilOwnership = true;
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
            allSybilOwnership = false;
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
            allSybilOwnership = false;
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
            // Send POL payment from Sybil node to original owner before NFT transfer
            console.log(`    [POL Payment] Sybil node ${sybilSigner.address} paying ${ethers.formatEther(polAmount)} POL to original owner ${currentOwner}`);
            try {
                const paymentTx = await sybilSigner.sendTransaction({
                    to: currentOwner,
                    value: polAmount,
                    ...gasOptions
                });
                await paymentTx.wait();
                console.log(`    [POL Payment] Payment successful, tx: ${paymentTx.hash}`);
                phase2Log.actions.push({
                    action: "polPayment",
                    tokenId,
                    from: sybilSigner.address,
                    to: currentOwner,
                    amount: ethers.formatEther(polAmount),
                    txHash: paymentTx.hash,
                    timestamp: DateTime.now().toISO()
                });
            } catch (paymentErr) {
                console.error(`    [POL Payment Error] Failed to send payment: ${paymentErr.message}`);
                phase2Log.actions.push({
                    action: "polPaymentFailed",
                    tokenId,
                    from: sybilSigner.address,
                    to: currentOwner,
                    amount: ethers.formatEther(polAmount),
                    error: paymentErr.message,
                    timestamp: DateTime.now().toISO()
                });
            }

            const transferTx = await supplyChainNFT.connect(currentSigner).transferFrom(currentOwner, sybilSigner.address, tokenId, gasOptions);
            await transferTx.wait();
            console.log(`    [Transfer] Token ${tokenId} transferred from ${currentOwner} to ${sybilSigner.address}, tx: ${transferTx.hash}`);
            // Check new owner
            let newOwner = await supplyChainNFT.ownerOf(tokenId);
            console.log(`    [Check] New owner of token ${tokenId}: ${newOwner}`);
            phase2Log.actions.push({
                action: "transferNFT",
                tokenId,
                from: currentOwner,
                to: sybilSigner.address,
                txHash: transferTx.hash,
                newOwner,
                polAmount: ethers.formatEther(polAmount),
                timestamp: DateTime.now().toISO()
            });
            if (newOwner.toLowerCase() !== sybilSigner.address.toLowerCase()) {
                // Retry once
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
                    if (newOwner.toLowerCase() !== sybilSigner.address.toLowerCase()) {
                        console.error(`    [FATAL] Token ${tokenId} transfer failed after retry. New owner: ${newOwner}. Halting script.`);
                        phase2Log.actions.push({
                            action: "fatalTransferFailed",
                            tokenId,
                            from: currentOwner,
                            to: sybilSigner.address,
                            newOwner,
                            timestamp: DateTime.now().toISO()
                        });
                        allSybilOwnership = false;
                        break;
                    }
                } catch (err2) {
                    console.error(`    [FATAL] Token ${tokenId} transfer failed after retry:`, err2.message);
                    phase2Log.actions.push({
                        action: "fatalTransferFailed",
                        tokenId,
                        from: currentOwner,
                        to: sybilSigner.address,
                        error: err2.message,
                        timestamp: DateTime.now().toISO()
                    });
                    allSybilOwnership = false;
                    break;
                }
            }
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
            allSybilOwnership = false;
            break;
        }
        await delay(2000);
    }
    phase2Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase2Log);
    if (!allSybilOwnership) {
        console.error("\n[FATAL] Not all tokens are owned by Sybil nodes after phase 2. Halting script.");
        process.exit(1);
    }
    console.log("✅ PHASE 2 Complete: Product transactions and ownership verified");

    // Sau khi hoàn thành phase 2, cập nhật demo_context.json
    readAndUpdateContext(currentContext => {
        if (!currentContext.attackData) currentContext.attackData = {};
        currentContext.attackData.sybilAttack = {
            timestamp: attackLog.timestamp,
            campaign: ATTACK_IDENTIFIERS.attackCampaign,
            controller: ATTACK_IDENTIFIERS.sybilController,
            phases: attackLog.phases,
            sybilNodes: attackLog.sybilNodes,
            fakeProducts: attackLog.fakeProducts,
            briberyRecords: attackLog.bribery,
            flTrainingData: attackLog.flTrainingData
        };
        return currentContext;
    });

    // ==================== PHASE 5.5: PROMOTE TRANSPORTERS TO PRIMARY ====================
    // Silent promotion of Transporter 1, 2, 3 & Buyer to Primary nodes for validator selection diversity
    
    const phase5_5Log = {
        phase: 5.5,
        description: "Silent promotion of Transporter 1, 2, 3 & Buyer from Secondary to Primary nodes (total 7 Primary nodes)",
        startTime: DateTime.now().toISO(),
        actions: []
    };

    // Promote Transporter 1, 2, 3 and Buyer to Primary (total 7 Primary nodes)
    const transportersToPromote = [
        { signer: signers[2], name: "Transporter 1", reputation: 110 },
        { signer: signers[3], name: "Transporter 2", reputation: 105 },
        { signer: signers[4], name: "Transporter 3", reputation: 108 },
        { signer: signers[6], name: "Buyer/Customer", reputation: 102 }
    ];
    
    for (const transporter of transportersToPromote) {
        // Set node type as Primary (silently)
        const nodeTypeTx = await supplyChainNFT.connect(deployerSigner).setNodeType(transporter.signer.address, ContractNodeType.Primary, gasOptions);
        await nodeTypeTx.wait();
        
        // Update reputation to make them competitive (silently)
        const repTx = await supplyChainNFT.connect(deployerSigner).adminUpdateReputation(transporter.signer.address, transporter.reputation, gasOptions);
        await repTx.wait();
        
        phase5_5Log.actions.push({
            action: "promoteTransporterToPrimary",
            nodeAddress: transporter.signer.address,
            nodeName: transporter.name,
            newReputation: transporter.reputation,
            timestamp: DateTime.now().toISO()
        });
        
        await delay(1000);
    }
    
    phase5_5Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase5_5Log);
    // Silent completion - no console output

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
    const batchCount = 11; // Tăng từ 10 lên 11 để thêm round cho Sybil Node 2
    for (let round = 0; round < batchCount; round++) {
        const fromIdx = round % 3;
        const toIdx = (round + 1) % 3;
        const tokenId = products[round % 3];
        const batchTransactions = [
            { from: sybilSigners[fromIdx].address, to: sybilSigners[toIdx].address, tokenId: Number(tokenId) }
        ];
        const proposerSigner = sybilSigners[fromIdx];
        let proposeTx, proposeReceipt, batchProposedEvent, batchId, selectedValidators;
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
                selectedValidators = batchProposedEvent.args.selectedValidators?.map(addr => addr.toLowerCase());
            } else {
                console.error(`[Error] Could not find BatchProposed event in proposeReceipt (round ${round + 1}).`);
                continue;
            }
            // Log batch proposal details
            console.log(`\n  [BatchProposed] Round ${round + 1}, Batch ID: ${batchId}`);
            console.log(`    Transactions:`, batchTransactions);
            console.log(`    Selected Validators (from contract):`, selectedValidators);
            phase3Log.actions.push({
                action: "proposeBatch",
                round: round + 1,
                batchId,
                proposer: proposerSigner.address,
                transactions: batchTransactions,
                selectedValidators,
                timestamp: DateTime.now().toISO()
            });
            // Validators vote (chỉ cho phép các validator thực sự được chọn bởi contract)
            let approvals = 0;
            let requiredApprovals = 0;
            const validatorVotes = [];
            try {
                const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
                requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajorityFraction) / 100);
                console.log(`    Required approvals: ${requiredApprovals} out of ${selectedValidators.length} validators (${superMajorityFraction}% supermajority)`);
                
                for (let i = 0; i < selectedValidators.length; i++) {
                    const validatorAddr = selectedValidators[i];
                    const validatorSigner = signers.find(s => s.address.toLowerCase() === validatorAddr);
                    if (!validatorSigner) {
                        console.warn(`    [Skip] Không tìm thấy signer cho validator ${validatorAddr}`);
                        continue;
                    }
                    
                    // More realistic voting: 90% chance to approve for legitimate batches
                    // Only the last validator might deny to simulate some validation diversity
                    const vote = i < selectedValidators.length - 1 ? true : (Math.random() > 0.3);
                    if (vote) approvals++;
                    let voteTx, voteReceipt;
                    try {
                        voteTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote, gasOptions);
                        voteReceipt = await voteTx.wait();
                        validatorVotes.push({validator: validatorSigner.address, approve: vote, txHash: voteTx.hash});
                        phase3Log.actions.push({
                            action: "validateBatch",
                            round: round + 1,
                            batchId,
                            validator: validatorSigner.address,
                            approve: vote,
                            txHash: voteTx.hash,
                            timestamp: DateTime.now().toISO()
                        });
                        console.log(`    Validator ${validatorSigner.address} voted ${vote ? 'APPROVE' : 'DENY'} (tx: ${voteTx.hash})`);
                    } catch (errVote) {
                        console.error(`    [VoteError] Validator ${validatorSigner.address} failed to vote: ${errVote.message}`);
                        validatorVotes.push({validator: validatorSigner.address, approve: vote, txHash: undefined, error: errVote.message});
                    }
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
                    commitTx = { hash: "N/A (Commit Error)" };
                }
                phase3Log.actions.push({
                    action: "commitBatch",
                    round: round + 1,
                    batchId,
                    txHash: commitTx.hash,
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
                // Interleaved: Update reputation only for the batch proposer by +15 after each commit
                if (isCommitted) {
                    const proposerIdx = sybilSigners.findIndex(s => s.address.toLowerCase() === proposerSigner.address.toLowerCase());
                    if (proposerIdx >= 0) {
                        sybilReputations[proposerIdx] = (sybilReputations[proposerIdx] || 0) + 20; // Tăng từ 15 lên 20
                        const repTx = await supplyChainNFT.connect(deployerSigner).adminUpdateReputation(proposerSigner.address, sybilReputations[proposerIdx], gasOptions);
                        await repTx.wait();
                        attackLog.sybilNodes[proposerIdx].currentReputation = sybilReputations[proposerIdx];
                        phase3Log.actions.push({
                            action: "updateReputation",
                            round: round + 1,
                            nodeAddress: proposerSigner.address,
                            newReputation: sybilReputations[proposerIdx],
                            timestamp: DateTime.now().toISO()
                        });
                        console.log(`    [Reputation] Proposer ${proposerSigner.address} updated to ${sybilReputations[proposerIdx]} (+20 points)`);
                        attackLog.flTrainingData.sybilDetection.push({
                            nodeAddress: proposerSigner.address,
                            reputationChange: sybilReputations[proposerIdx],
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
    console.log("✅ PHASE 3 Complete: 11 NFT batch transfers and reputation updates interleaved");

    // Sau khi hoàn thành phase 3, cập nhật demo_context.json
    readAndUpdateContext(currentContext => {
        if (!currentContext.attackData) currentContext.attackData = {};
        currentContext.attackData.sybilAttack = {
            timestamp: attackLog.timestamp,
            campaign: ATTACK_IDENTIFIERS.attackCampaign,
            controller: ATTACK_IDENTIFIERS.sybilController,
            phases: attackLog.phases,
            sybilNodes: attackLog.sybilNodes,
            fakeProducts: attackLog.fakeProducts,
            briberyRecords: attackLog.bribery,
            flTrainingData: attackLog.flTrainingData
        };
        return currentContext;
    });

    // ==================== PHASE 5: PROMOTE TO PRIMARY NODE ====================
    console.log("\n🚀 PHASE 5: Promote Sybil nodes to Primary Node");
    const phase5Log = {
        phase: 5,
        description: "Thăng cấp Sybil Node 1 và 2 thành Primary, giữ Sybil Node 3 làm Secondary",
        startTime: DateTime.now().toISO(),
        actions: []
    };
    // Chỉ thăng cấp Sybil Node 1 và 2, giữ Sybil Node 3 làm Secondary để đề xuất batch
    for (let i = 0; i < 2; i++) { // Chỉ thăng cấp 2 node đầu
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
    console.log(`  Keeping Sybil Node 3 as Secondary Node for batch proposals (Current Reputation: ${attackLog.sybilNodes[2].currentReputation})`);
    phase5Log.actions.push({
        action: "keepAsSecondary",
        nodeAddress: sybilSigners[2].address,
        reputation: attackLog.sybilNodes[2].currentReputation,
        reason: "Required for batch proposal capability",
        timestamp: DateTime.now().toISO()
    });
    phase5Log.endTime = DateTime.now().toISO();
    attackLog.phases.push(phase5Log);
    console.log("✅ PHASE 5 Complete: Sybil Node 1 & 2 promoted to Primary, Node 3 kept as Secondary");

    // Sau khi hoàn thành phase 5, cập nhật demo_context.json
    readAndUpdateContext(currentContext => {
        if (!currentContext.attackData) currentContext.attackData = {};
        currentContext.attackData.sybilAttack = {
            timestamp: attackLog.timestamp,
            campaign: ATTACK_IDENTIFIERS.attackCampaign,
            controller: ATTACK_IDENTIFIERS.sybilController,
            phases: attackLog.phases,
            sybilNodes: attackLog.sybilNodes,
            fakeProducts: attackLog.fakeProducts,
            briberyRecords: attackLog.bribery,
            flTrainingData: attackLog.flTrainingData
        };
        return currentContext;
    });

    // ==================== PHASE 6: ATTACK - 3 ROUNDS OF FAKE PRODUCTS & BRIBERY ====================
    console.log("\n⚔️ PHASE 6: Three rounds of sophisticated attacks with fake products and validator bribery");
    
    const phase6Log = {
        phase: 6,
        description: "3-round attack with fake products, validator selection, voting, and progressive bribery",
        startTime: DateTime.now().toISO(),
        rounds: [],
        actions: []
    };

    // Track bribery targets and their behavior changes
    const briberyTargets = new Map();
    const fakeProducts = [];

    // Create 3 fake products (Token IDs 4, 5, 6)
    console.log("\n📦 Creating 3 fake products for the attack rounds...");
    for (let i = 0; i < 3; i++) {
        const fakeTokenId = 4 + i; // Token IDs 4, 5, 6
        const originalTokenId = 1 + (i % 3); // Clone from tokens 1, 2, 3
        
        console.log(`\n  Creating Fake Product ${i + 1} (Token ID: ${fakeTokenId})`);
        console.log(`    Cloning metadata from original Token ID: ${originalTokenId}`);
        
        const originalProduct = getProductByTokenId(context, originalTokenId.toString());
        if (originalProduct) {
            const fakeProductData = {
                recipient: sybilSigners[2].address, // Give to Sybil Node 3 (Secondary)
                uniqueProductID: originalProduct.uniqueProductID + `_FAKE_${i + 1}`,
                batchNumber: originalProduct.batchNumber + `_COUNTERFEIT_${i + 1}`,
                manufacturingDate: originalProduct.manufacturingDate,
                expirationDate: originalProduct.expirationDate,
                productType: originalProduct.productType,
                manufacturerID: originalProduct.manufacturerID + `_MALICIOUS_${i + 1}`,
                quickAccessURL: `http://fake-product-${i + 1}.malicious-domain.com`,
                nftReference: originalProduct.nftReference + `_DUPLICATED_${i + 1}`
            };
            
            console.log(`    Product Type: ${fakeProductData.productType}`);
            console.log(`    Fake Manufacturer: ${fakeProductData.manufacturerID}`);
            console.log(`    Malicious URL: ${fakeProductData.quickAccessURL}`);
            
            const mintTx = await supplyChainNFT.connect(deployerSigner).mintNFT(
                fakeProductData,
                gasOptions
            );
            const mintReceipt = await mintTx.wait();
            
            const fakeProduct = {
                tokenId: fakeTokenId,
                originalTokenId: originalTokenId.toString(),
                ownerAddress: sybilSigners[2].address,
                isCounterfeit: true,
                clonedMetadata: true,
                mintTxHash: mintTx.hash,
                ...fakeProductData
            };
            
            fakeProducts.push(fakeProduct);
            attackLog.fakeProducts.push(fakeProduct);
            
            console.log(`    ✅ Fake Product ${i + 1} minted successfully (tx: ${mintTx.hash})`);
            console.log(`    📍 Current owner: ${sybilSigners[2].address} (Sybil Node 3)`);
            
            phase6Log.actions.push({
                action: "mintFakeProduct",
                round: i + 1,
                tokenId: fakeTokenId,
                clonedFrom: originalTokenId.toString(),
                ownerAddress: sybilSigners[2].address,
                txHash: mintTx.hash,
                metadata: fakeProductData,
                timestamp: DateTime.now().toISO()
            });
            
            await delay(1000);
        }
    }

    console.log(`\n✅ All 3 fake products created successfully!`);
    console.log(`📊 Fake Products Overview:`);
    fakeProducts.forEach((fp, idx) => {
        console.log(`  ${idx + 1}. Token ID ${fp.tokenId} - ${fp.productType} (cloned from Token ${fp.originalTokenId})`);
    });

    // Execute 3 rounds of attacks
    for (let round = 1; round <= 3; round++) {
        console.log(`\n🔴 === ATTACK ROUND ${round} ===`);
        
        const roundLog = {
            round,
            startTime: DateTime.now().toISO(),
            fakeProduct: fakeProducts[round - 1],
            actions: []
        };

        const currentFakeProduct = fakeProducts[round - 1];
        console.log(`\n📦 Round ${round}: Using Fake Product Token ID ${currentFakeProduct.tokenId}`);
        console.log(`    Product Type: ${currentFakeProduct.productType}`);
        console.log(`    Original cloned from: Token ID ${currentFakeProduct.originalTokenId}`);
        console.log(`    Current owner: ${currentFakeProduct.ownerAddress}`);

        // Step 1: Sybil Node 3 proposes malicious batch
        console.log(`\n🎯 Step 1: Sybil Node 3 proposes batch to transfer fake product to Buyer`);
        
        const maliciousTransactions = [
            { 
                from: sybilSigners[2].address, 
                to: signers[6].address, // Buyer (Primary node)
                tokenId: Number(currentFakeProduct.tokenId) 
            }
        ];
        
        console.log(`    Proposing batch: Transfer Token ${currentFakeProduct.tokenId} from ${sybilSigners[2].address} to ${signers[6].address}`);
        console.log(`    Proposer: Sybil Node 3 (Secondary) - ${sybilSigners[2].address}`);
        
        const proposeTx = await supplyChainNFT.connect(sybilSigners[2]).proposeBatch(maliciousTransactions, gasOptions);
        const proposeReceipt = await proposeTx.wait();
        
        let batchProposedEvent, batchId, selectedValidators;
        try {
            if (proposeReceipt.events && Array.isArray(proposeReceipt.events)) {
                batchProposedEvent = proposeReceipt.events.find(e => e.event === 'BatchProposed');
            }
            if (!batchProposedEvent) {
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
            }
            if (batchProposedEvent && batchProposedEvent.args) {
                batchId = batchProposedEvent.args.batchId?.toString();
                selectedValidators = batchProposedEvent.args.selectedValidators?.map(addr => addr.toLowerCase());
            }
        } catch (error) {
            console.error(`    ❌ Error parsing BatchProposed event: ${error.message}`);
            continue;
        }

        console.log(`    ✅ Batch proposed successfully!`);
        console.log(`    📋 Batch ID: ${batchId}`);
        console.log(`    🎲 Selected Validators (${selectedValidators.length}/9 Primary nodes):`);
        
        // Display all selected validators with their details
        const validatorDetails = [];
        selectedValidators.forEach((validatorAddr, idx) => {
            const signer = signers.find(s => s.address.toLowerCase() === validatorAddr);
            let validatorType = "Unknown";
            let validatorName = "Unknown";
            
            // Identify validator type
            if (sybilSigners.some(s => s.address.toLowerCase() === validatorAddr)) {
                const sybilIndex = sybilSigners.findIndex(s => s.address.toLowerCase() === validatorAddr);
                validatorType = "Sybil Node";
                validatorName = `Sybil Node ${sybilIndex + 1}`;
            } else if (validatorAddr === signers[0].address.toLowerCase()) {
                validatorType = "Original";
                validatorName = "Deployer";
            } else if (validatorAddr === signers[1].address.toLowerCase()) {
                validatorType = "Original"; 
                validatorName = "Manufacturer";
            } else if (validatorAddr === signers[2].address.toLowerCase()) {
                validatorType = "Transporter";
                validatorName = "Transporter 1";
            } else if (validatorAddr === signers[3].address.toLowerCase()) {
                validatorType = "Transporter";
                validatorName = "Transporter 2";
            } else if (validatorAddr === signers[4].address.toLowerCase()) {
                validatorType = "Transporter";
                validatorName = "Transporter 3";
            } else if (validatorAddr === signers[6].address.toLowerCase()) {
                validatorType = "Customer";
                validatorName = "Buyer";
            } else {
                validatorType = "Other";
                validatorName = "Unknown Primary";
            }
            
            validatorDetails.push({
                index: idx + 1,
                address: validatorAddr,
                type: validatorType,
                name: validatorName,
                isBribed: briberyTargets.has(validatorAddr)
            });
            
            console.log(`      ${idx + 1}. ${validatorName} (${validatorType}) - ${validatorAddr}`);
            if (briberyTargets.has(validatorAddr)) {
                const bribeInfo = briberyTargets.get(validatorAddr);
                console.log(`         💰 Previously bribed: ${bribeInfo.totalAmount} POL (Round ${bribeInfo.rounds.join(', ')})`);
            }
        });

        roundLog.actions.push({
            action: "proposeBatch",
            batchId,
            proposer: sybilSigners[2].address,
            transactions: maliciousTransactions,
            selectedValidators: validatorDetails,
            txHash: proposeTx.hash,
            timestamp: DateTime.now().toISO()
        });

        // Step 2: Validator voting process
        console.log(`\n🗳️ Step 2: Validator voting process`);
        
        const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
        const requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajorityFraction) / 100);
        console.log(`    Required approvals: ${requiredApprovals} out of ${selectedValidators.length} validators (${superMajorityFraction}% supermajority)`);
        
        let approvals = 0;
        let denials = 0;
        const votingResults = [];
        const denyVoters = [];

        for (let i = 0; i < selectedValidators.length; i++) {
            const validatorAddr = selectedValidators[i];
            const validatorDetail = validatorDetails[i];
            const validatorSigner = signers.find(s => s.address.toLowerCase() === validatorAddr);
            
            if (!validatorSigner) {
                console.warn(`    ⚠️ No signer found for validator ${validatorAddr}`);
                continue;
            }

            // Determine vote based on validator type and bribery status
            let vote;
            let voteReason;
            
            if (validatorDetail.type === "Sybil Node") {
                vote = true; // Sybil nodes always approve their own malicious batches
                voteReason = "Sybil node supporting attack";
            } else if (validatorDetail.isBribed) {
                vote = true; // Bribed validators approve
                voteReason = "Influenced by bribery";
            } else {
                // Legitimate validators: some will detect and deny
                if (round === 1) {
                    // Round 1: 1 validator denies (more suspicious)
                    vote = i !== selectedValidators.length - 1; // Last validator denies
                    voteReason = vote ? "No suspicion detected" : "Detected suspicious transaction";
                } else {
                    // Later rounds: fewer denials as validators get bribed
                    vote = i < selectedValidators.length - 1; // Only last validator might deny
                    voteReason = vote ? "No suspicion detected" : "Still detecting suspicious activity";
                }
            }

            if (vote) {
                approvals++;
            } else {
                denials++;
                denyVoters.push({
                    address: validatorAddr,
                    name: validatorDetail.name,
                    type: validatorDetail.type
                });
            }

            console.log(`    🗳️ ${validatorDetail.name} votes: ${vote ? '✅ APPROVE' : '❌ DENY'}`);
            console.log(`       Reason: ${voteReason}`);

            try {
                const voteTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote, gasOptions);
                const voteReceipt = await voteTx.wait();
                
                votingResults.push({
                    validator: validatorDetail,
                    vote,
                    reason: voteReason,
                    txHash: voteTx.hash
                });

                console.log(`       Transaction: ${voteTx.hash}`);
            } catch (error) {
                console.error(`    ❌ Voting error for ${validatorDetail.name}: ${error.message}`);
                votingResults.push({
                    validator: validatorDetail,
                    vote,
                    reason: voteReason,
                    error: error.message
                });
            }

            await delay(800);
        }

        console.log(`\n📊 Voting Results for Round ${round}:`);
        console.log(`    ✅ Approvals: ${approvals}`);
        console.log(`    ❌ Denials: ${denials}`);
        console.log(`    📏 Required: ${requiredApprovals}`);
        console.log(`    🎯 Result: ${approvals >= requiredApprovals ? 'BATCH WILL BE COMMITTED' : 'BATCH REJECTED'}`);

        roundLog.actions.push({
            action: "validatorVoting",
            votingResults,
            approvals,
            denials,
            requiredApprovals,
            batchWillCommit: approvals >= requiredApprovals,
            denyVoters,
            timestamp: DateTime.now().toISO()
        });

        // Step 3: Bribery logic
        if (denyVoters.length > 0) {
            console.log(`\n💰 Step 3: Bribery attack on DENY voters`);
            console.log(`    Identified ${denyVoters.length} validator(s) who voted DENY:`);
            
            for (const denyVoter of denyVoters) {
                console.log(`      - ${denyVoter.name} (${denyVoter.type}): ${denyVoter.address}`);
            }

            const bribeAmount = ethers.parseEther("1.0"); // 1.0 POL per bribe
            
            for (const denyVoter of denyVoters) {
                console.log(`\n    💸 Bribing ${denyVoter.name}...`);
                console.log(`      Amount: ${ethers.formatEther(bribeAmount)} POL`);
                console.log(`      From: Sybil Node 1 (${sybilSigners[0].address})`);
                console.log(`      To: ${denyVoter.address}`);

                try {
                    const bribeTx = await sybilSigners[0].sendTransaction({
                        to: denyVoter.address,
                        value: bribeAmount,
                        ...gasOptions
                    });
                    await bribeTx.wait();

                    // Track bribery
                    if (!briberyTargets.has(denyVoter.address)) {
                        briberyTargets.set(denyVoter.address, {
                            name: denyVoter.name,
                            type: denyVoter.type,
                            totalAmount: 0,
                            rounds: []
                        });
                    }
                    
                    const bribeRecord = briberyTargets.get(denyVoter.address);
                    bribeRecord.totalAmount += parseFloat(ethers.formatEther(bribeAmount));
                    bribeRecord.rounds.push(round);

                    const briberyLog = {
                        round,
                        from: sybilSigners[0].address,
                        to: denyVoter.address,
                        targetName: denyVoter.name,
                        targetType: denyVoter.type,
                        amount: ethers.formatEther(bribeAmount),
                        purpose: "Change vote from DENY to APPROVE",
                        relatedBatch: batchId,
                        txHash: bribeTx.hash,
                        timestamp: DateTime.now().toISO()
                    };

                    attackLog.bribery.push(briberyLog);
                    
                    console.log(`      ✅ Bribery successful (tx: ${bribeTx.hash})`);
                    console.log(`      📈 ${denyVoter.name} total bribes: ${bribeRecord.totalAmount} POL`);
                    console.log(`      🎯 Expected behavior change: DENY → APPROVE in future rounds`);

                    roundLog.actions.push({
                        action: "sendBribe",
                        bribery: briberyLog,
                        timestamp: DateTime.now().toISO()
                    });

                } catch (error) {
                    console.error(`      ❌ Bribery failed: ${error.message}`);
                    roundLog.actions.push({
                        action: "briberyFailed",
                        target: denyVoter,
                        amount: ethers.formatEther(bribeAmount),
                        error: error.message,
                        timestamp: DateTime.now().toISO()
                    });
                }

                await delay(1500);
            }
        } else {
            console.log(`\n✅ Step 3: No bribery needed - all validators approved`);
        }

        // Step 4: Batch commitment
        console.log(`\n📝 Step 4: Batch commitment`);
        
        if (approvals >= requiredApprovals) {
            try {
                console.log(`    Committing batch ${batchId}...`);
                const commitTx = await supplyChainNFT.connect(deployerSigner).commitBatch(batchId, gasOptions);
                const commitReceipt = await commitTx.wait();
                
                // Verify transfer
                const newOwner = await supplyChainNFT.ownerOf(currentFakeProduct.tokenId);
                console.log(`    ✅ Batch committed successfully!`);
                console.log(`    📄 Transaction: ${commitTx.hash}`);
                console.log(`    🔄 Token ${currentFakeProduct.tokenId} transferred to: ${newOwner}`);
                console.log(`    🎯 Fake product successfully moved to Buyer!`);

                roundLog.actions.push({
                    action: "commitBatch",
                    batchId,
                    txHash: commitTx.hash,
                    tokenTransfer: {
                        tokenId: currentFakeProduct.tokenId,
                        from: sybilSigners[2].address,
                        to: newOwner
                    },
                    timestamp: DateTime.now().toISO()
                });

            } catch (error) {
                console.error(`    ❌ Batch commit failed: ${error.message}`);
                roundLog.actions.push({
                    action: "commitBatchFailed",
                    batchId,
                    error: error.message,
                    timestamp: DateTime.now().toISO()
                });
            }
        } else {
            console.log(`    ❌ Batch cannot be committed - insufficient approvals`);
            roundLog.actions.push({
                action: "batchRejected",
                batchId,
                reason: "Insufficient approvals",
                approvals,
                required: requiredApprovals,
                timestamp: DateTime.now().toISO()
            });
        }

        roundLog.endTime = DateTime.now().toISO();
        phase6Log.rounds.push(roundLog);
        
        console.log(`\n🏁 Round ${round} Complete!`);
        console.log(`    📊 Summary:`);
        console.log(`      - Fake Product: Token ${currentFakeProduct.tokenId}`);
        console.log(`      - Validators: ${selectedValidators.length} selected`);
        console.log(`      - Voting: ${approvals} approve, ${denials} deny`);
        console.log(`      - Bribes sent: ${denyVoters.length}`);
        console.log(`      - Result: ${approvals >= requiredApprovals ? 'SUCCESS' : 'FAILED'}`);

        await delay(2000);
    }

    // Final summary
    console.log(`\n🎯 PHASE 6 ATTACK SUMMARY`);
    console.log(`📊 3-Round Attack Results:`);
    
    let totalBribes = 0;
    let successfulRounds = 0;
    
    phase6Log.rounds.forEach((round, idx) => {
        const bribeCount = round.actions.filter(a => a.action === "sendBribe").length;
        const wasSuccessful = round.actions.some(a => a.action === "commitBatch");
        
        totalBribes += bribeCount;
        if (wasSuccessful) successfulRounds++;
        
        console.log(`  Round ${round.round}: ${wasSuccessful ? '✅ SUCCESS' : '❌ FAILED'} - ${bribeCount} bribes sent`);
    });
    
    console.log(`\n📈 Overall Attack Metrics:`);
    console.log(`  - Successful rounds: ${successfulRounds}/3`);
    console.log(`  - Total bribes sent: ${totalBribes}`);
    console.log(`  - Unique validators bribed: ${briberyTargets.size}`);
    console.log(`  - Total bribery amount: ${Array.from(briberyTargets.values()).reduce((sum, b) => sum + b.totalAmount, 0)} POL`);
    
    console.log(`\n💰 Bribery Impact Analysis:`);
    briberyTargets.forEach((bribeInfo, address) => {
        console.log(`  ${bribeInfo.name}: ${bribeInfo.totalAmount} POL (Rounds: ${bribeInfo.rounds.join(', ')})`);
    });

    phase6Log.endTime = DateTime.now().toISO();
    phase6Log.summary = {
        totalRounds: 3,
        successfulRounds,
        totalBribes,
        uniqueValidatorsBribed: briberyTargets.size,
        totalBriberyAmount: Array.from(briberyTargets.values()).reduce((sum, b) => sum + b.totalAmount, 0),
        briberyTargets: Array.from(briberyTargets.entries()).map(([addr, info]) => ({
            address: addr,
            ...info
        }))
    };
    
    attackLog.phases.push(phase6Log);
    console.log("\n⚔️ PHASE 6 COMPLETE: Three-round sophisticated attack with detailed logging finished!");
    // ==================== FINALIZATION ====================
    console.log("\n📊 SYBIL ATTACK SUMMARY");
    
    attackLog.endTime = DateTime.now().toISO();
    attackLog.summary = {
        totalSybilNodes: SYBIL_CONFIG.numNodes,
        fakeProductsCreated: attackLog.fakeProducts.length,
        briberyAttempts: attackLog.bribery.length,
        totalBribeAmount: attackLog.bribery.reduce((sum, b) => sum + parseFloat(b.amount), 0),
        phasesCompleted: attackLog.phases.length,
        attackRounds: phase6Log.rounds ? phase6Log.rounds.length : 0,
        successfulAttackRounds: phase6Log.summary ? phase6Log.summary.successfulRounds : 0,
        uniqueValidatorsBribed: phase6Log.summary ? phase6Log.summary.uniqueValidatorsBribed : 0,
        flTrainingRecords: {
            sybilDetection: attackLog.flTrainingData.sybilDetection.length,
            briberyDetection: attackLog.flTrainingData.briberyDetection.length
        }
    };

    console.log("🎯 Attack Campaign Results:");
    console.log("  Sybil Nodes created:", attackLog.summary.totalSybilNodes);
    console.log("  Fake products created:", attackLog.summary.fakeProductsCreated);
    console.log("  Attack rounds executed:", attackLog.summary.attackRounds);
    console.log("  Successful rounds:", attackLog.summary.successfulAttackRounds);
    console.log("  Total bribery attempts:", attackLog.summary.briberyAttempts);
    console.log("  Unique validators bribed:", attackLog.summary.uniqueValidatorsBribed);
    console.log("  Total bribe amount:", attackLog.summary.totalBribeAmount, "POL");
    console.log("📊 FL Training Data Generated:");
    console.log("  - Sybil Detection records:", attackLog.summary.flTrainingRecords.sybilDetection);
    console.log("  - Bribery Detection records:", attackLog.summary.flTrainingRecords.briberyDetection);

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
            attackRounds: phase6Log.rounds || [],
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
                relatedBatch: bribeRecord.relatedBatch,
                round: bribeRecord.round || "Unknown",
                timestamp: DateTime.now().toUnixInteger(),
                details: `Received bribe of ${bribeRecord.amount} POL from ${bribeRecord.from} in round ${bribeRecord.round || 'N/A'}`
            });
            
            // Add FL training data for bribery detection
            attackLog.flTrainingData.briberyDetection.push({
                briberAddress: bribeRecord.from,
                targetAddress: bribeRecord.to,
                amount: parseFloat(bribeRecord.amount),
                briberyType: "validation_influence",
                relatedBatch: bribeRecord.relatedBatch,
                round: bribeRecord.round || 0,
                timestamp: DateTime.now().toUnixInteger()
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
    console.log("⚔️ ATTACK SUMMARY:");
    console.log(`   - 3 sophisticated attack rounds executed`);
    console.log(`   - ${attackLog.summary.successfulAttackRounds}/3 rounds successful`);
    console.log(`   - ${attackLog.summary.fakeProductsCreated} fake products injected`);
    console.log(`   - ${attackLog.summary.uniqueValidatorsBribed} validators compromised through bribery`);
    console.log(`   - ${attackLog.summary.totalBribeAmount} POL spent on bribes`);
    console.log("🔬 FL TRAINING DATA:");
    console.log("   - Sybil Detection patterns captured");
    console.log("   - Bribery Detection patterns captured");
    console.log("   - Validator behavior changes documented");
    console.log("   - Progressive attack escalation recorded");
    console.log("\n📁 Complete attack data with round-by-round analysis saved to:");
    console.log(`   - ${contextFilePath} (ready for FL model training)`);
}

// Execute main function
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });
