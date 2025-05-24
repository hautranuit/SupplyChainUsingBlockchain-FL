const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

// --- Helper Function Definition ---
const contextFilePath = path.join(__dirname, 'demo_context.json'); // Centralized data file for FL

function readAndUpdateContext(updateFn) {
    let contextData = {};
    if (fs.existsSync(contextFilePath)) {
        try {
            const fileContent = fs.readFileSync(contextFilePath, 'utf8');
            contextData = fileContent.trim() === "" ? {} : JSON.parse(fileContent);
        } catch (error) {
            console.error(`Error reading or parsing ${contextFilePath}:`, error);
            contextData = {}; // Start fresh on error
        }
    }

    const updatedContext = updateFn(contextData);

    try {
        fs.writeFileSync(contextFilePath, JSON.stringify(updatedContext, null, 2));
        console.log(`Context data updated successfully in ${contextFilePath}`);
    } catch (error) {
        console.error(`Error writing context data to ${contextFilePath}:`, error);
    }
    return updatedContext;
}

// Helper function to simulate a delay
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
// --- End Helper Function ---

async function main() {
    console.log("--- Starting 04: Transport and Finalization Scenario (Fixed Token ID Lookup) ---");

    // Load context
    let currentContext = {};
    if (fs.existsSync(contextFilePath)) {
        try {
            currentContext = JSON.parse(fs.readFileSync(contextFilePath, 'utf8'));
        } catch (error) {
            console.error(`Error reading initial context from ${contextFilePath}:`, error);
            process.exit(1);
        }
    }

    const contractAddress = currentContext.contractAddress;
    const nodes = currentContext.nodes;
    const products = currentContext.products; // Assuming products is an object keyed by tokenId

    if (!contractAddress) {
        console.error("Error: Contract address not found in context.");
        process.exit(1);
    }
     if (!nodes) {
        console.error("Error: Nodes information not found in context.");
        process.exit(1);
    }
     if (!products || Object.keys(products).length === 0) {
        console.error("Error: Products information not found or empty in context.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();

    // Find accounts based on context
    const manufacturerAcc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Manufacturer"));
    const transporter1Acc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Transporter 1"));
    const transporter2Acc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Transporter 2"));
    const transporter3Acc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Transporter 3 (Batch Proposer)"));
    const retailerAcc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Retailer"));
    const buyer1Acc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Buyer/Customer"));

    if (!manufacturerAcc || !transporter1Acc || !transporter2Acc || !transporter3Acc || !retailerAcc || !buyer1Acc) {
        console.error("Error: Could not find one or more required accounts based on context.");
        process.exit(1);
    }

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress);
    console.log("Connected to contract SupplyChainNFT.");

    // Define gas options (adjust if needed)
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'),
        maxFeePerGas: ethers.parseUnits('40', 'gwei')
    };

    // --- Identify Products and Token IDs Dynamically from Context ---
    console.log("\nLooking up Token IDs from context based on uniqueProductID...");
    const product1 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_001");
    const product2 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_002");
    const product3 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_003");

    if (!product1 || !product1.tokenId) {
        console.error("Error: Could not find product DEMO_PROD_001 or its tokenId in context.");
        process.exit(1);
    }
    const tokenId1 = product1.tokenId;
    console.log(`  Found DEMO_PROD_001 -> Token ID: ${tokenId1}`);

    if (!product2 || !product2.tokenId) {
        console.error("Error: Could not find product DEMO_PROD_002 or its tokenId in context.");
        process.exit(1);
    }
    const tokenId2 = product2.tokenId;
    console.log(`  Found DEMO_PROD_002 -> Token ID: ${tokenId2}`);

    if (!product3 || !product3.tokenId) {
        console.error("Error: Could not find product DEMO_PROD_003 or its tokenId in context.");
        process.exit(1);
    }
    const tokenId3 = product3.tokenId;
    console.log(`  Found DEMO_PROD_003 -> Token ID: ${tokenId3}`);

    // --- Scenario 1: Product 1 (Token ID: ${tokenId1}) - Sequential Transport & Finalization ---
    console.log(`\n--- Scenario 1: Product 1 (Token ID: ${tokenId1}) ---`);

    if (product1.status !== "CollateralDeposited") {
        console.error(`Product 1 (Token ID: ${tokenId1}) is not in 'CollateralDeposited' state. Current state: ${product1.status}. Aborting transport.`);
        process.exit(1);
    }

    console.log("\n--- Product 1: Sequential Transport and Finalization ---");
    let currentOwnerP1Signer = manufacturerAcc; // Manufacturer should still own the NFT at this point
    let currentOwnerP1Address = manufacturerAcc.address.toLowerCase();

    // Verify owner on-chain matches context before starting transport
    try {
        const ownerOnChain = await supplyChainNFT.ownerOf(tokenId1);
        if (ownerOnChain.toLowerCase() !== currentOwnerP1Address) {
             console.warn(`  WARN: Owner mismatch for Token ID ${tokenId1}. Context: ${currentOwnerP1Address}, On-Chain: ${ownerOnChain.toLowerCase()}. Proceeding based on context owner.`);
             // Optionally, find the signer for the on-chain owner if needed, but for starting transport, manufacturer should act.
        }
    } catch (ownerError) {
        console.error(`  ERROR checking owner for Token ID ${tokenId1}:`, ownerError);
        process.exit(1);
    }

    // 1.1 Manufacturer starts transport for Product 1
    console.log(`  1.1 Manufacturer (${manufacturerAcc.address}) starting transport for Token ID ${tokenId1}.`);
    const p1_transportLegs = [transporter1Acc.address, transporter2Acc.address, transporter3Acc.address];
    const p1_startLocation = "Manufacturer Site";
    const p1_endLocation = "Buyer 1 Location";
    const p1_distance = 450;
    let startTx1, startReceipt1, startBlock1;
    try {
        startTx1 = await supplyChainNFT.connect(manufacturerAcc).startTransport(tokenId1, p1_transportLegs, p1_startLocation, p1_endLocation, p1_distance, gasOptions);
        startReceipt1 = await startTx1.wait(1);
        startBlock1 = await ethers.provider.getBlock(startReceipt1.blockNumber);
        console.log(`      TransportStarted event emitted. Gas: ${startReceipt1.gasUsed.toString()}.`);
    } catch (error) {
        console.error(`  ERROR calling startTransport for Token ID ${tokenId1}:`, error);
        process.exit(1);
    }

    // Update context after starting transport
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx; // Safety check
        p.status = "InTransit";
        p.transportInfo = {
            legs: p1_transportLegs.map(addr => addr.toLowerCase()),
            startLocation: p1_startLocation,
            endLocation: p1_endLocation,
            totalDistance: p1_distance,
            currentLegIndex: 0,
            currentTransporter: transporter1Acc.address.toLowerCase(),
            updates: []
        };
        p.lastUpdateTimestamp = startBlock1.timestamp;
        if (!p.history) p.history = [];
        p.history.push({
            event: "TransportStarted",
            actor: manufacturerAcc.address.toLowerCase(),
            timestamp: startBlock1.timestamp,
            details: `Transport started. Legs: ${p1_transportLegs.join(' -> ')}. From: ${p1_startLocation}, To: ${p1_endLocation}, Dist: ${p1_distance}km.`,
            txHash: startReceipt1.transactionHash
        });
        if (ctx.nodes && ctx.nodes[manufacturerAcc.address.toLowerCase()]) {
             if (!ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions) ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions = [];
            ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({
                type: "StartTransport", tokenId: tokenId1, timestamp: startBlock1.timestamp,
                details: `Started transport for product ${tokenId1}.`, txHash: startReceipt1.transactionHash
            });
        }
        return ctx;
    });

    // 1.2 Manufacturer transfers NFT to Transporter 1
    console.log(`      Manufacturer (${manufacturerAcc.address}) transferring Token ID ${tokenId1} to Transporter 1 (${transporter1Acc.address}).`);
    let transferTx_M_T1, transferReceipt_M_T1, transferBlock_M_T1;
    try {
        transferTx_M_T1 = await supplyChainNFT.connect(manufacturerAcc).transferFrom(manufacturerAcc.address, transporter1Acc.address, tokenId1, gasOptions);
        transferReceipt_M_T1 = await transferTx_M_T1.wait(1);
        transferBlock_M_T1 = await ethers.provider.getBlock(transferReceipt_M_T1.blockNumber);
        console.log(`      NFT ${tokenId1} transferred to Transporter 1. Gas: ${transferReceipt_M_T1.gasUsed.toString()}.`);
        currentOwnerP1Signer = transporter1Acc;
        currentOwnerP1Address = transporter1Acc.address.toLowerCase();
    } catch (error) {
        console.error(`  ERROR transferring NFT ${tokenId1} from Manufacturer to Transporter 1:`, error);
        process.exit(1);
    }

    // Update context after transfer
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx;
        p.currentOwnerAddress = currentOwnerP1Address;
        p.lastUpdateTimestamp = transferBlock_M_T1.timestamp;
        if (!p.history) p.history = [];
        p.history.push({
            event: "Transfer", actor: manufacturerAcc.address.toLowerCase(), from: manufacturerAcc.address.toLowerCase(), to: currentOwnerP1Address,
            timestamp: transferBlock_M_T1.timestamp, details: `NFT transferred to transporter 1.`, txHash: transferReceipt_M_T1.transactionHash
        });
        if (ctx.nodes && ctx.nodes[manufacturerAcc.address.toLowerCase()]) {
             if (!ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions) ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions = [];
            ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({ type: "TransferNFT", tokenId: tokenId1, to: currentOwnerP1Address, timestamp: transferBlock_M_T1.timestamp, txHash: transferReceipt_M_T1.transactionHash });
        }
        if (ctx.nodes && ctx.nodes[currentOwnerP1Address]) {
             if (!ctx.nodes[currentOwnerP1Address].interactions) ctx.nodes[currentOwnerP1Address].interactions = [];
            ctx.nodes[currentOwnerP1Address].interactions.push({ type: "ReceiveNFT", tokenId: tokenId1, from: manufacturerAcc.address.toLowerCase(), timestamp: transferBlock_M_T1.timestamp, txHash: transferReceipt_M_T1.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 1.3 Transporter 1 transfers to Transporter 2
    console.log(`  1.3 Transporter 1 (${transporter1Acc.address}) transferring Token ID ${tokenId1} to Transporter 2 (${transporter2Acc.address}).`);
    let transferTx_T1_T2, transferReceipt_T1_T2, transferBlock_T1_T2;
    let previousOwnerP1Address_T1 = currentOwnerP1Address;
    try {
        transferTx_T1_T2 = await supplyChainNFT.connect(transporter1Acc).transferFrom(transporter1Acc.address, transporter2Acc.address, tokenId1, gasOptions);
        transferReceipt_T1_T2 = await transferTx_T1_T2.wait(1);
        transferBlock_T1_T2 = await ethers.provider.getBlock(transferReceipt_T1_T2.blockNumber);
        console.log(`      NFT ${tokenId1} transferred to Transporter 2. Gas: ${transferReceipt_T1_T2.gasUsed.toString()}.`);
        currentOwnerP1Signer = transporter2Acc;
        currentOwnerP1Address = transporter2Acc.address.toLowerCase();
    } catch (error) {
        console.error(`  ERROR transferring NFT ${tokenId1} from Transporter 1 to Transporter 2:`, error);
        process.exit(1);
    }

    // Update context
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx;
        p.currentOwnerAddress = currentOwnerP1Address;
        if (p.transportInfo) p.transportInfo.currentLegIndex = 1;
        p.lastUpdateTimestamp = transferBlock_T1_T2.timestamp;
        if (!p.history) p.history = [];
        p.history.push({ event: "Transfer", actor: previousOwnerP1Address_T1, from: previousOwnerP1Address_T1, to: currentOwnerP1Address, timestamp: transferBlock_T1_T2.timestamp, details: `NFT transferred to transporter 2.`, txHash: transferReceipt_T1_T2.transactionHash });
        if (ctx.nodes && ctx.nodes[previousOwnerP1Address_T1]) {
             if (!ctx.nodes[previousOwnerP1Address_T1].interactions) ctx.nodes[previousOwnerP1Address_T1].interactions = [];
            ctx.nodes[previousOwnerP1Address_T1].interactions.push({ type: "TransferNFT", tokenId: tokenId1, to: currentOwnerP1Address, timestamp: transferBlock_T1_T2.timestamp, txHash: transferReceipt_T1_T2.transactionHash });
        }
        if (ctx.nodes && ctx.nodes[currentOwnerP1Address]) {
             if (!ctx.nodes[currentOwnerP1Address].interactions) ctx.nodes[currentOwnerP1Address].interactions = [];
            ctx.nodes[currentOwnerP1Address].interactions.push({ type: "ReceiveNFT", tokenId: tokenId1, from: previousOwnerP1Address_T1, timestamp: transferBlock_T1_T2.timestamp, txHash: transferReceipt_T1_T2.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 1.4 Transporter 2 transfers to Transporter 3
    console.log(`  1.4 Transporter 2 (${transporter2Acc.address}) transferring Token ID ${tokenId1} to Transporter 3 (${transporter3Acc.address}).`);
    let transferTx_T2_T3, transferReceipt_T2_T3, transferBlock_T2_T3;
    let previousOwnerP1Address_T2 = currentOwnerP1Address;
    try {
        transferTx_T2_T3 = await supplyChainNFT.connect(transporter2Acc).transferFrom(transporter2Acc.address, transporter3Acc.address, tokenId1, gasOptions);
        transferReceipt_T2_T3 = await transferTx_T2_T3.wait(1);
        transferBlock_T2_T3 = await ethers.provider.getBlock(transferReceipt_T2_T3.blockNumber);
        console.log(`      NFT ${tokenId1} transferred to Transporter 3. Gas: ${transferReceipt_T2_T3.gasUsed.toString()}.`);
        currentOwnerP1Signer = transporter3Acc;
        currentOwnerP1Address = transporter3Acc.address.toLowerCase();
    } catch (error) {
        console.error(`  ERROR transferring NFT ${tokenId1} from Transporter 2 to Transporter 3:`, error);
        process.exit(1);
    }

    // Update context
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx;
        p.currentOwnerAddress = currentOwnerP1Address;
        if (p.transportInfo) p.transportInfo.currentLegIndex = 2;
        p.lastUpdateTimestamp = transferBlock_T2_T3.timestamp;
        if (!p.history) p.history = [];
        p.history.push({ event: "Transfer", actor: previousOwnerP1Address_T2, from: previousOwnerP1Address_T2, to: currentOwnerP1Address, timestamp: transferBlock_T2_T3.timestamp, details: `NFT transferred to transporter 3.`, txHash: transferReceipt_T2_T3.transactionHash });
        if (ctx.nodes && ctx.nodes[previousOwnerP1Address_T2]) {
             if (!ctx.nodes[previousOwnerP1Address_T2].interactions) ctx.nodes[previousOwnerP1Address_T2].interactions = [];
            ctx.nodes[previousOwnerP1Address_T2].interactions.push({ type: "TransferNFT", tokenId: tokenId1, to: currentOwnerP1Address, timestamp: transferBlock_T2_T3.timestamp, txHash: transferReceipt_T2_T3.transactionHash });
        }
        if (ctx.nodes && ctx.nodes[currentOwnerP1Address]) {
             if (!ctx.nodes[currentOwnerP1Address].interactions) ctx.nodes[currentOwnerP1Address].interactions = [];
            ctx.nodes[currentOwnerP1Address].interactions.push({ type: "ReceiveNFT", tokenId: tokenId1, from: previousOwnerP1Address_T2, timestamp: transferBlock_T2_T3.timestamp, txHash: transferReceipt_T2_T3.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 1.5 Transporter 3 completes final transport leg
    console.log(`  1.5 Transporter 3 (${transporter3Acc.address}) completing final transport leg for Token ID ${tokenId1}.`);
    let completeTx1, completeReceipt1, completeBlock1;
    try {
        completeTx1 = await supplyChainNFT.connect(transporter3Acc).completeTransport(tokenId1, gasOptions);
        completeReceipt1 = await completeTx1.wait(1);
        completeBlock1 = await ethers.provider.getBlock(completeReceipt1.blockNumber);
        console.log(`      TransportCompleted event emitted. Gas: ${completeReceipt1.gasUsed.toString()}.`);
    } catch (error) {
        console.error(`  ERROR completing transport for Token ID ${tokenId1}:`, error);
        process.exit(1);
    }

    // Update context after completing transport
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx;
        p.status = "TransportCompleted";
        p.lastUpdateTimestamp = completeBlock1.timestamp;
        if (!p.history) p.history = [];
        p.history.push({
            event: "TransportCompleted", actor: transporter3Acc.address.toLowerCase(), timestamp: completeBlock1.timestamp,
            details: `Final transport leg completed.`, txHash: completeReceipt1.transactionHash
        });
        if (ctx.nodes && ctx.nodes[transporter3Acc.address.toLowerCase()]) {
             if (!ctx.nodes[transporter3Acc.address.toLowerCase()].interactions) ctx.nodes[transporter3Acc.address.toLowerCase()].interactions = [];
            ctx.nodes[transporter3Acc.address.toLowerCase()].interactions.push({
                type: "CompleteTransport", tokenId: tokenId1, timestamp: completeBlock1.timestamp,
                details: `Completed final transport leg for product ${tokenId1}.`, txHash: completeReceipt1.transactionHash
            });
        }
        return ctx;
    });
    await delay(1000);

    // 1.6 Buyer confirms delivery and finalizes purchase for Product 1
    console.log(`  1.6 Buyer 1 (${buyer1Acc.address}) confirming delivery and finalizing purchase for Product 1 (Token ID: ${tokenId1}).`);
    const meetsIncentiveCriteriaP1 = true; // Example

    // Log pre-finalization state from context
    console.log("    --- Pre-Finalization State (from context) ---");
    const p1Ctx = currentContext.products[tokenId1];
    if (!p1Ctx) {
        console.error(`  FATAL: Product 1 (Token ID: ${tokenId1}) not found in context before finalization.`);
        process.exit(1);
    }
    console.log(`    Token ID: ${tokenId1}`);
    console.log(`    Seller: ${p1Ctx.sellerAddress}`);
    console.log(`    Buyer: ${p1Ctx.pendingBuyerAddress}`);
    console.log(`    Current Owner (Context): ${p1Ctx.currentOwnerAddress}`); // Should be Transporter 3
    console.log(`    Price: ${ethers.formatEther(p1Ctx.price || '0')} ETH`);
    console.log(`    Collateral: ${ethers.formatEther(p1Ctx.collateralAmount || '0')} ETH`);
    console.log(`    Status: ${p1Ctx.status}`);
    console.log("    --- End Pre-Finalization State ---");

    let finalizeTx1, finalizeReceipt1, finalizeBlock1;
    try {
        finalizeTx1 = await supplyChainNFT.connect(buyer1Acc).confirmDeliveryAndFinalize(tokenId1, meetsIncentiveCriteriaP1, gasOptions);
        finalizeReceipt1 = await finalizeTx1.wait(1);
        finalizeBlock1 = await ethers.provider.getBlock(finalizeReceipt1.blockNumber);
        console.log(`      PurchaseFinalized event emitted. Gas: ${finalizeReceipt1.gasUsed.toString()}.`);
        console.log(`      NFT ${tokenId1} should now be owned by Buyer 1 (${buyer1Acc.address}).`);
    } catch (error) {
        console.error(`  ERROR finalizing purchase for Token ID ${tokenId1}:`, error);
        process.exit(1);
    }

    // Update context after finalization
    currentContext = readAndUpdateContext(ctx => {
        const p = ctx.products[tokenId1];
        if (!p) return ctx;
        p.status = "DeliveredToBuyer";
        p.currentOwnerAddress = buyer1Acc.address.toLowerCase(); // NFT transferred to buyer
        p.lastUpdateTimestamp = finalizeBlock1.timestamp;
        if (!p.history) p.history = [];
        p.history.push({
            event: "PurchaseFinalized", actor: buyer1Acc.address.toLowerCase(), timestamp: finalizeBlock1.timestamp,
            details: `Buyer confirmed delivery. Meets Incentive: ${meetsIncentiveCriteriaP1}. NFT transferred. Payment released.`, txHash: finalizeReceipt1.transactionHash
        });
        if (ctx.nodes && ctx.nodes[buyer1Acc.address.toLowerCase()]) {
             if (!ctx.nodes[buyer1Acc.address.toLowerCase()].interactions) ctx.nodes[buyer1Acc.address.toLowerCase()].interactions = [];
            ctx.nodes[buyer1Acc.address.toLowerCase()].interactions.push({
                type: "FinalizePurchase", tokenId: tokenId1, timestamp: finalizeBlock1.timestamp,
                details: `Confirmed delivery for product ${tokenId1}.`, txHash: finalizeReceipt1.transactionHash
            });
        }
        // Also record NFT reception for buyer
        if (ctx.nodes && ctx.nodes[buyer1Acc.address.toLowerCase()]) {
             if (!ctx.nodes[buyer1Acc.address.toLowerCase()].interactions) ctx.nodes[buyer1Acc.address.toLowerCase()].interactions = [];
            // Find the previous owner from history or context if needed, assume transporter3 for now
            const previousOwner = transporter3Acc.address.toLowerCase(); 
            ctx.nodes[buyer1Acc.address.toLowerCase()].interactions.push({ type: "ReceiveNFT", tokenId: tokenId1, from: previousOwner, timestamp: finalizeBlock1.timestamp, txHash: finalizeReceipt1.transactionHash });
        }
        return ctx;
    });

    // --- Scenario 2 & 3: Product 2 & 3 (Token IDs: ${tokenId2}, ${tokenId3}) - Batched Transport & Finalization ---
    console.log(`\n--- Scenario 2 & 3: Products 2 & 3 (Token IDs: ${tokenId2}, ${tokenId3}) - Batched Transport & Finalization ---`);

    // 2.1 Manufacturer starts transport for Product 2 & 3
    const p2_transportLegs = [transporter1Acc.address, transporter2Acc.address, retailerAcc.address];
    const p3_transportLegs = [transporter1Acc.address, transporter2Acc.address, retailerAcc.address];
    const p2_startLocation = "Manufacturer's Bulk Warehouse";
    const p2_endLocation = "Retailer Central Warehouse";
    const p2_distance = 700;
    const p3_startLocation = "Manufacturer's Bulk Warehouse";
    const p3_endLocation = "Retailer Central Warehouse";
    const p3_distance = 720;
    let startTx2, startReceipt2, startBlock2, startTx3, startReceipt3, startBlock3;
    try {
        startTx2 = await supplyChainNFT.connect(manufacturerAcc).startTransport(tokenId2, p2_transportLegs, p2_startLocation, p2_endLocation, p2_distance, gasOptions);
        startReceipt2 = await startTx2.wait(1);
        startBlock2 = await ethers.provider.getBlock(startReceipt2.blockNumber);
        startTx3 = await supplyChainNFT.connect(manufacturerAcc).startTransport(tokenId3, p3_transportLegs, p3_startLocation, p3_endLocation, p3_distance, gasOptions);
        startReceipt3 = await startTx3.wait(1);
        startBlock3 = await ethers.provider.getBlock(startReceipt3.blockNumber);
        console.log(`      TransportStarted for both products.`);
    } catch (error) {
        console.error(`  ERROR calling startTransport for Token IDs ${tokenId2}, ${tokenId3}:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.status = "InTransit";
            p2.transportInfo = {
                legs: p2_transportLegs.map(addr => addr.toLowerCase()),
                startLocation: p2_startLocation,
                endLocation: p2_endLocation,
                totalDistance: p2_distance,
                currentLegIndex: 0,
                currentTransporter: transporter1Acc.address.toLowerCase(),
                updates: []
            };
            p2.lastUpdateTimestamp = startBlock2.timestamp;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "TransportStarted", actor: manufacturerAcc.address.toLowerCase(), timestamp: startBlock2.timestamp, details: `Transport started. Legs: ${p2_transportLegs.join(' -> ')}.`, txHash: startReceipt2.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.status = "InTransit";
            p3.transportInfo = {
                legs: p3_transportLegs.map(addr => addr.toLowerCase()),
                startLocation: p3_startLocation,
                endLocation: p3_endLocation,
                totalDistance: p3_distance,
                currentLegIndex: 0,
                currentTransporter: transporter1Acc.address.toLowerCase(),
                updates: []
            };
            p3.lastUpdateTimestamp = startBlock3.timestamp;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "TransportStarted", actor: manufacturerAcc.address.toLowerCase(), timestamp: startBlock3.timestamp, details: `Transport started. Legs: ${p3_transportLegs.join(' -> ')}.`, txHash: startReceipt3.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 2.2 Manufacturer transfers both NFTs to Transporter 1
    let transferTx_M_T1_2, transferReceipt_M_T1_2, transferBlock_M_T1_2, transferTx_M_T1_3, transferReceipt_M_T1_3, transferBlock_M_T1_3;
    try {
        transferTx_M_T1_2 = await supplyChainNFT.connect(manufacturerAcc).transferFrom(manufacturerAcc.address, transporter1Acc.address, tokenId2, gasOptions);
        transferReceipt_M_T1_2 = await transferTx_M_T1_2.wait(1);
        transferBlock_M_T1_2 = await ethers.provider.getBlock(transferReceipt_M_T1_2.blockNumber);
        transferTx_M_T1_3 = await supplyChainNFT.connect(manufacturerAcc).transferFrom(manufacturerAcc.address, transporter1Acc.address, tokenId3, gasOptions);
        transferReceipt_M_T1_3 = await transferTx_M_T1_3.wait(1);
        transferBlock_M_T1_3 = await ethers.provider.getBlock(transferReceipt_M_T1_3.blockNumber);
        console.log(`      Both NFTs transferred to Transporter 1.`);
    } catch (error) {
        console.error(`  ERROR transferring NFTs to Transporter 1:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.currentOwnerAddress = transporter1Acc.address.toLowerCase();
            p2.lastUpdateTimestamp = transferBlock_M_T1_2.timestamp;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "Transfer", actor: manufacturerAcc.address.toLowerCase(), from: manufacturerAcc.address.toLowerCase(), to: transporter1Acc.address.toLowerCase(), timestamp: transferBlock_M_T1_2.timestamp, details: `NFT transferred to transporter 1.`, txHash: transferReceipt_M_T1_2.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.currentOwnerAddress = transporter1Acc.address.toLowerCase();
            p3.lastUpdateTimestamp = transferBlock_M_T1_3.timestamp;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "Transfer", actor: manufacturerAcc.address.toLowerCase(), from: manufacturerAcc.address.toLowerCase(), to: transporter1Acc.address.toLowerCase(), timestamp: transferBlock_M_T1_3.timestamp, details: `NFT transferred to transporter 1.`, txHash: transferReceipt_M_T1_3.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 2.3 Transporter 1 proposes batch transfer to Transporter 2
    const batchTxData_T1_T2 = [
        { from: transporter1Acc.address, to: transporter2Acc.address, tokenId: tokenId2 },
        { from: transporter1Acc.address, to: transporter2Acc.address, tokenId: tokenId3 }
    ];
    let proposeBatchTx1, proposeBatchReceipt1, batchId_T1_T2;
    try {
        proposeBatchTx1 = await supplyChainNFT.connect(transporter1Acc).proposeBatch(batchTxData_T1_T2, gasOptions);
        proposeBatchReceipt1 = await proposeBatchTx1.wait(1);
        const proposeEvent_T1_T2_log = proposeBatchReceipt1.logs.find(log => log.fragment && log.fragment.name === "BatchProposed");
        if (!proposeEvent_T1_T2_log) throw new Error("BatchProposed event not found for T1->T2 batch");
        batchId_T1_T2 = proposeEvent_T1_T2_log.args.batchId;
        console.log(`      Batch ${batchId_T1_T2} proposed by Transporter 1.`);
    } catch (error) {
        console.error(`  ERROR proposing batch from T1 to T2:`, error);
        process.exit(1);
    }
    await delay(1000);

    // 2.4 Validators validate the batch (simulate Manufacturer and Retailer as validators)
    try {
        await supplyChainNFT.connect(manufacturerAcc).validateBatch(batchId_T1_T2, true, gasOptions);
        await supplyChainNFT.connect(retailerAcc).validateBatch(batchId_T1_T2, true, gasOptions);
        console.log(`      Validators validated batch ${batchId_T1_T2}.`);
    } catch (error) {
        console.error(`  ERROR validating batch ${batchId_T1_T2}:`, error);
        process.exit(1);
    }
    await delay(1000);

    // 2.5 Transporter 1 commits the batch (NFTs move to Transporter 2)
    let commitBatchTx1, commitBatchReceipt1;
    try {
        commitBatchTx1 = await supplyChainNFT.connect(transporter1Acc).commitBatch(batchId_T1_T2, gasOptions);
        commitBatchReceipt1 = await commitBatchTx1.wait(1);
        console.log(`      Batch ${batchId_T1_T2} committed by Transporter 1.`);
    } catch (error) {
        console.error(`  ERROR committing batch ${batchId_T1_T2}:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.currentOwnerAddress = transporter2Acc.address.toLowerCase();
            if (p2.transportInfo) p2.transportInfo.currentLegIndex = 1;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "BatchTransfer", actor: transporter1Acc.address.toLowerCase(), from: transporter1Acc.address.toLowerCase(), to: transporter2Acc.address.toLowerCase(), details: `Batch transfer to transporter 2.`, txHash: commitBatchReceipt1.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.currentOwnerAddress = transporter2Acc.address.toLowerCase();
            if (p3.transportInfo) p3.transportInfo.currentLegIndex = 1;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "BatchTransfer", actor: transporter1Acc.address.toLowerCase(), from: transporter1Acc.address.toLowerCase(), to: transporter2Acc.address.toLowerCase(), details: `Batch transfer to transporter 2.`, txHash: commitBatchReceipt1.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 2.6 Transporter 2 proposes batch transfer to Retailer
    const batchTxData_T2_Retailer = [
        { from: transporter2Acc.address, to: retailerAcc.address, tokenId: tokenId2 },
        { from: transporter2Acc.address, to: retailerAcc.address, tokenId: tokenId3 }
    ];
    let proposeBatchTx2, proposeBatchReceipt2, batchId_T2_Retailer;
    try {
        proposeBatchTx2 = await supplyChainNFT.connect(transporter2Acc).proposeBatch(batchTxData_T2_Retailer, gasOptions);
        proposeBatchReceipt2 = await proposeBatchTx2.wait(1);
        const proposeEvent_T2_Retailer_log = proposeBatchReceipt2.logs.find(log => log.fragment && log.fragment.name === "BatchProposed");
        if (!proposeEvent_T2_Retailer_log) throw new Error("BatchProposed event not found for T2->Retailer batch");
        batchId_T2_Retailer = proposeEvent_T2_Retailer_log.args.batchId;
        console.log(`      Batch ${batchId_T2_Retailer} proposed by Transporter 2.`);
    } catch (error) {
        console.error(`  ERROR proposing batch from T2 to Retailer:`, error);
        process.exit(1);
    }
    await delay(1000);

    // 2.7 Validators validate the batch
    try {
        await supplyChainNFT.connect(manufacturerAcc).validateBatch(batchId_T2_Retailer, true, gasOptions);
        await supplyChainNFT.connect(retailerAcc).validateBatch(batchId_T2_Retailer, true, gasOptions);
        console.log(`      Validators validated batch ${batchId_T2_Retailer}.`);
    } catch (error) {
        console.error(`  ERROR validating batch ${batchId_T2_Retailer}:`, error);
        process.exit(1);
    }
    await delay(1000);

    // 2.8 Transporter 2 commits the batch (NFTs move to Retailer)
    let commitBatchTx2, commitBatchReceipt2;
    try {
        commitBatchTx2 = await supplyChainNFT.connect(transporter2Acc).commitBatch(batchId_T2_Retailer, gasOptions);
        commitBatchReceipt2 = await commitBatchTx2.wait(1);
        console.log(`      Batch ${batchId_T2_Retailer} committed by Transporter 2.`);
    } catch (error) {
        console.error(`  ERROR committing batch ${batchId_T2_Retailer}:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.currentOwnerAddress = retailerAcc.address.toLowerCase();
            if (p2.transportInfo) p2.transportInfo.currentLegIndex = 2;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "BatchTransfer", actor: transporter2Acc.address.toLowerCase(), from: transporter2Acc.address.toLowerCase(), to: retailerAcc.address.toLowerCase(), details: `Batch transfer to retailer.`, txHash: commitBatchReceipt2.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.currentOwnerAddress = retailerAcc.address.toLowerCase();
            if (p3.transportInfo) p3.transportInfo.currentLegIndex = 2;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "BatchTransfer", actor: transporter2Acc.address.toLowerCase(), from: transporter2Acc.address.toLowerCase(), to: retailerAcc.address.toLowerCase(), details: `Batch transfer to retailer.`, txHash: commitBatchReceipt2.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 2.9 Retailer completes transport for both products
    let completeTx2, completeReceipt2, completeBlock2, completeTx3, completeReceipt3, completeBlock3;
    try {
        completeTx2 = await supplyChainNFT.connect(retailerAcc).completeTransport(tokenId2, gasOptions);
        completeReceipt2 = await completeTx2.wait(1);
        completeBlock2 = await ethers.provider.getBlock(completeReceipt2.blockNumber);
        completeTx3 = await supplyChainNFT.connect(retailerAcc).completeTransport(tokenId3, gasOptions);
        completeReceipt3 = await completeTx3.wait(1);
        completeBlock3 = await ethers.provider.getBlock(completeReceipt3.blockNumber);
        console.log(`      Retailer completed transport for both products.`);
    } catch (error) {
        console.error(`  ERROR completing transport for Token IDs ${tokenId2}, ${tokenId3}:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.status = "TransportCompleted";
            p2.lastUpdateTimestamp = completeBlock2.timestamp;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "TransportCompleted", actor: retailerAcc.address.toLowerCase(), timestamp: completeBlock2.timestamp, details: `Retailer completed final transport leg.`, txHash: completeReceipt2.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.status = "TransportCompleted";
            p3.lastUpdateTimestamp = completeBlock3.timestamp;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "TransportCompleted", actor: retailerAcc.address.toLowerCase(), timestamp: completeBlock3.timestamp, details: `Retailer completed final transport leg.`, txHash: completeReceipt3.transactionHash });
        }
        return ctx;
    });
    await delay(1000);

    // 2.10 Retailer finalizes purchase for both products
    const meetsIncentiveCriteriaP2 = true;
    const meetsIncentiveCriteriaP3 = true;
    let finalizeTx2, finalizeReceipt2, finalizeBlock2, finalizeTx3, finalizeReceipt3, finalizeBlock3;
    try {
        finalizeTx2 = await supplyChainNFT.connect(retailerAcc).confirmDeliveryAndFinalize(tokenId2, meetsIncentiveCriteriaP2, gasOptions);
        finalizeReceipt2 = await finalizeTx2.wait(1);
        finalizeBlock2 = await ethers.provider.getBlock(finalizeReceipt2.blockNumber);
        finalizeTx3 = await supplyChainNFT.connect(retailerAcc).confirmDeliveryAndFinalize(tokenId3, meetsIncentiveCriteriaP3, gasOptions);
        finalizeReceipt3 = await finalizeTx3.wait(1);
        finalizeBlock3 = await ethers.provider.getBlock(finalizeReceipt3.blockNumber);
        console.log(`      Retailer finalized purchase for both products.`);
    } catch (error) {
        console.error(`  ERROR finalizing purchase for Token IDs ${tokenId2}, ${tokenId3}:`, error);
        process.exit(1);
    }
    // Update context for both products
    currentContext = readAndUpdateContext(ctx => {
        const p2 = ctx.products[tokenId2];
        if (p2) {
            p2.status = "DeliveredToBuyer";
            p2.currentOwnerAddress = retailerAcc.address.toLowerCase();
            p2.lastUpdateTimestamp = finalizeBlock2.timestamp;
            if (!p2.history) p2.history = [];
            p2.history.push({ event: "PurchaseFinalized", actor: retailerAcc.address.toLowerCase(), timestamp: finalizeBlock2.timestamp, details: `Retailer confirmed delivery. NFT transferred. Payment released.`, txHash: finalizeReceipt2.transactionHash });
        }
        const p3 = ctx.products[tokenId3];
        if (p3) {
            p3.status = "DeliveredToBuyer";
            p3.currentOwnerAddress = retailerAcc.address.toLowerCase();
            p3.lastUpdateTimestamp = finalizeBlock3.timestamp;
            if (!p3.history) p3.history = [];
            p3.history.push({ event: "PurchaseFinalized", actor: retailerAcc.address.toLowerCase(), timestamp: finalizeBlock3.timestamp, details: `Retailer confirmed delivery. NFT transferred. Payment released.`, txHash: finalizeReceipt3.transactionHash });
        }
        return ctx;
    });
    await delay(1000);


    console.log("\n--- 04: Transport and Finalization Scenario Complete ---");
    console.log("Key outcomes:");
    console.log(`  - Product 1 (ID: ${tokenId1}): Transported via 3 legs, finalized by Buyer 1 (${buyer1Acc.address}).`);
    console.log(`  - Product 2 (ID: ${tokenId2}): Transferred directly, finalized by Retailer (${retailerAcc.address}).`);
    console.log(`  - Product 3 (ID: ${tokenId3}): Transferred directly, finalized by Retailer (${retailerAcc.address}).`);
    console.log("  Next step: 05_scenario_batch_processing.cjs");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });

