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
// --- End Helper Function ---

async function main() {
    console.log("--- Starting 05: Batch Processing Scenarios - Product Exchange (Fixed Token ID Lookup) ---");

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
    const deployer = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Deployer/Admin"));
    const retailerAcc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Retailer"));
    const buyer1Acc = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Buyer/Customer")); // Buyer1 will be the batch proposer

    if (!deployer || !retailerAcc || !buyer1Acc) {
        console.error("Error: Could not find deployer, retailer, or buyer1 account based on context.");
        process.exit(1);
    }
    const batchProposerAcc = buyer1Acc;

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract SupplyChainNFT.");

    // Define gas options (adjust if needed)
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'),
        maxFeePerGas: ethers.parseUnits('40', 'gwei')
    };

    // --- Product Exchange Simulation via Batch ---
    console.log("\n--- Product Exchange Simulation via Batch ---");
    console.log(`Proposer (Buyer1): ${batchProposerAcc.address}`);
    console.log(`Counterparty (Retailer): ${retailerAcc.address}`);

    // Find products and Token IDs dynamically from context
    console.log("\nLooking up Token IDs from context based on uniqueProductID...");
    const product1 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_001"); // Should be owned by Buyer1 after script 04
    const product2 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_002"); // Should be owned by Retailer after script 04
    const product3 = Object.values(products).find(p => p.uniqueProductID === "DEMO_PROD_003"); // Should be owned by Retailer after script 04

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

    // Verify initial ownership from context (based on script 04 completion)
    if (product1.currentOwnerAddress.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
        console.error(`Error: Product 1 (Token ID: ${tokenId1}) owner mismatch. Expected ${buyer1Acc.address} (Buyer1), found ${product1.currentOwnerAddress}`);
        process.exit(1);
    }
    if (product2.currentOwnerAddress.toLowerCase() !== retailerAcc.address.toLowerCase()) {
        console.error(`Error: Product 2 (Token ID: ${tokenId2}) owner mismatch. Expected ${retailerAcc.address} (Retailer), found ${product2.currentOwnerAddress}`);
        process.exit(1);
    }
     if (product3.currentOwnerAddress.toLowerCase() !== retailerAcc.address.toLowerCase()) {
        console.error(`Error: Product 3 (Token ID: ${tokenId3}) owner mismatch. Expected ${retailerAcc.address} (Retailer), found ${product3.currentOwnerAddress}`);
        process.exit(1);
    }

    // Define the batch transactions using the dynamically found Token IDs
    const transactionsForExchangeBatch = [
        { from: buyer1Acc.address, to: retailerAcc.address, tokenId: BigInt(tokenId1) },
        { from: retailerAcc.address, to: buyer1Acc.address, tokenId: BigInt(tokenId2) },
        { from: retailerAcc.address, to: buyer1Acc.address, tokenId: BigInt(tokenId3) }
    ];

    // --- Log rationale for batching and transaction details (giống script cũ) ---
    console.log("\nRationale for Batching these transactions:");
    console.log("  Instead of executing 3 separate NFT transfer transactions (each incurring gas fees and requiring individual confirmation), we are batching them.");
    console.log("  This approach offers several advantages:");
    console.log("    - Optimized Gas Costs: A single 'proposeBatch' and 'commitBatch' operation (plus validator votes) is generally more gas-efficient than multiple individual 'transferFrom' calls, especially as the number of batched items grows.");
    console.log("    - Improved Time Efficiency: Waiting for confirmations for fewer batch-level transactions is faster than for many individual ones.");
    console.log("    - Enhanced Atomicity: The entire exchange (all 3 transfers) either succeeds together or fails together if the batch isn't committed. This prevents partial exchanges, which is crucial for such multi-party swaps.");
    console.log("\nDetailed transactions for the exchange batch:");
    console.log(`  1. Product ${product2.uniqueProductID} (Token ID: ${tokenId2}) from Retailer (${retailerAcc.address}) to Buyer1 (${buyer1Acc.address})`);
    console.log(`  2. Product ${product3.uniqueProductID} (Token ID: ${tokenId3}) from Retailer (${retailerAcc.address}) to Buyer1 (${buyer1Acc.address})`);
    console.log(`  3. Product ${product1.uniqueProductID} (Token ID: ${tokenId1}) from Buyer1 (${buyer1Acc.address}) to Retailer (${retailerAcc.address})`);

    // Propose Batch
    let proposeTx, proposeReceipt, proposeBlock, proposeEvent, batchId, selectedValidators;
    try {
        proposeTx = await supplyChainNFT.connect(batchProposerAcc).proposeBatch(transactionsForExchangeBatch, gasOptions);
        proposeReceipt = await proposeTx.wait(1);
        proposeBlock = await ethers.provider.getBlock(proposeReceipt.blockNumber);
        console.log(`  Batch proposed for exchange. Gas Used: ${proposeReceipt.gasUsed.toString()}`);

        proposeEvent = proposeReceipt.logs?.map(log => {
            try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
        }).find(event => event?.name === "BatchProposed");

        if (!proposeEvent) throw new Error("BatchProposed event not found.");
        batchId = proposeEvent.args.batchId.toString();
        selectedValidators = proposeEvent.args.selectedValidators.map(addr => addr.toLowerCase());
        console.log(`  Batch ID: ${batchId}, Selected Validators (${selectedValidators.length}): ${selectedValidators.join(", ")}`);
    } catch (error) {
        console.error("  ERROR proposing batch:", error);
        process.exit(1);
    }

    // Update context with proposed batch
    currentContext = readAndUpdateContext(ctx => {
        if (!ctx.batches) ctx.batches = {};
        ctx.batches[batchId] = {
            batchId: batchId,
            proposer: batchProposerAcc.address.toLowerCase(),
            transactions: transactionsForExchangeBatch.map(t => ({ ...t, tokenId: t.tokenId.toString() })), // Store tokenIds as strings
            selectedValidators: selectedValidators,
            votes: {},
            status: "Proposed",
            proposeTimestamp: proposeBlock.timestamp,
            proposeTxHash: proposeReceipt.transactionHash
        };
        if (ctx.nodes && ctx.nodes[batchProposerAcc.address.toLowerCase()]) {
             if (!ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions) ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions = [];
            ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions.push({
                type: "ProposeBatch", batchId: batchId, timestamp: proposeBlock.timestamp,
                details: `Proposed exchange batch with ${transactionsForExchangeBatch.length} transactions.`, txHash: proposeReceipt.transactionHash
            });
        }
        return ctx;
    });

    // Validators Vote
    console.log("\n  Validators voting for the exchange batch...");
    let approvals = 0;
    let requiredApprovals = 0;
    const voteReceipts = {};
    try {
        const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
        requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajorityFraction) / 100);
        console.log(`    Supermajority: ${superMajorityFraction.toString()}%, Required Approvals: ${requiredApprovals}`);

        for (let i = 0; i < selectedValidators.length; i++) {
            const validatorAddr = selectedValidators[i];
            const validatorSigner = signers.find(s => s.address.toLowerCase() === validatorAddr);
            if (!validatorSigner) {
                console.warn(`    Could not find signer for validator ${validatorAddr}. Skipping vote.`);
                continue;
            }
            const vote = approvals < requiredApprovals; // Simulate approval until majority reached
            if (vote) approvals++;

            console.log(`    Validator ${validatorSigner.address} voting ${vote ? 'Approve' : 'Deny'}...`);
            let voteTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote, gasOptions);
            let voteReceipt = await voteTx.wait(1);
            let voteBlock = await ethers.provider.getBlock(voteReceipt.blockNumber);
            console.log(`      Vote cast. Gas Used: ${voteReceipt.gasUsed.toString()}`);
            voteReceipts[validatorAddr] = { receipt: voteReceipt, block: voteBlock, vote: vote };
        }
    } catch (error) {
        console.error("  ERROR during validator voting:", error);
        // Decide if we should proceed or exit. For now, log and continue to context update.
    }

    // Update context with votes
    currentContext = readAndUpdateContext(ctx => {
        const batch = ctx.batches[batchId];
        if (!batch) return ctx;
        let maxTimestamp = batch.proposeTimestamp; // Initialize with propose time
        for (const validatorAddr in voteReceipts) {
            const voteData = voteReceipts[validatorAddr];
            batch.votes[validatorAddr] = {
                vote: voteData.vote,
                timestamp: voteData.block.timestamp,
                txHash: voteData.receipt.transactionHash
            };
            maxTimestamp = Math.max(maxTimestamp, voteData.block.timestamp);
            if (ctx.nodes && ctx.nodes[validatorAddr]) {
                 if (!ctx.nodes[validatorAddr].interactions) ctx.nodes[validatorAddr].interactions = [];
                ctx.nodes[validatorAddr].interactions.push({
                    type: "VoteBatch", batchId: batchId, vote: voteData.vote, timestamp: voteData.block.timestamp,
                    details: `Voted ${voteData.vote ? 'Approve' : 'Deny'} on batch ${batchId}.`, txHash: voteData.receipt.transactionHash
                });
            }
        }
        // Determine status based on actual votes vs required
        const actualApprovals = Object.values(batch.votes).filter(v => v.vote).length;
        batch.status = actualApprovals >= requiredApprovals ? "VotingCompleted_Approved" : "VotingCompleted_Rejected";
        batch.lastUpdateTimestamp = maxTimestamp;
        return ctx;
    });

    // Commit Batch
    if (approvals >= requiredApprovals) {
        console.log("\n  Attempting to commit batch...");
        let commitTx, commitReceipt, commitBlock, isCommitted = false, isFlagged = false;
        try {
            commitTx = await supplyChainNFT.connect(batchProposerAcc).commitBatch(batchId, gasOptions);
            commitReceipt = await commitTx.wait(1);
            commitBlock = await ethers.provider.getBlock(commitReceipt.blockNumber);
            console.log(`    Batch commit attempted. Gas Used: ${commitReceipt.gasUsed.toString()}`);

            const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
            isCommitted = batchDetails.committed;
            isFlagged = batchDetails.flagged;
        } catch (error) {
            console.error("  ERROR attempting to commit batch:", error);
            // Assume commit failed if error occurs
            commitReceipt = { transactionHash: "N/A (Commit Error)" }; // Placeholder for context update
            commitBlock = { timestamp: Math.floor(Date.now() / 1000) }; // Approximate time
        }

        // Update context with commit result
        currentContext = readAndUpdateContext(ctx => {
            const batch = ctx.batches[batchId];
            if (!batch) return ctx;
            batch.status = isCommitted ? "Committed" : (isFlagged ? "Flagged" : "CommitFailed");
            batch.commitTimestamp = commitBlock.timestamp;
            batch.commitTxHash = commitReceipt.transactionHash;
            batch.lastUpdateTimestamp = commitBlock.timestamp;

            if (ctx.nodes && ctx.nodes[batchProposerAcc.address.toLowerCase()]) {
                 if (!ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions) ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions = [];
                ctx.nodes[batchProposerAcc.address.toLowerCase()].interactions.push({
                    type: "CommitBatchAttempt", batchId: batchId, status: batch.status, timestamp: commitBlock.timestamp,
                    details: `Attempted to commit batch ${batchId}. Result: ${batch.status}.`, txHash: commitReceipt.transactionHash
                });
            }

            // If committed, update product owners and history
            if (isCommitted) {
                transactionsForExchangeBatch.forEach(txDetail => {
                    const tokenIdStr = txDetail.tokenId.toString();
                    const product = ctx.products[tokenIdStr];
                    if (product) {
                        const oldOwner = product.currentOwnerAddress;
                        const newOwner = txDetail.to.toLowerCase();
                        product.currentOwnerAddress = newOwner;
                        product.status = `ExchangedInBatch_${batchId}`;
                        product.lastUpdateTimestamp = commitBlock.timestamp;
                        if (!product.history) product.history = [];
                        product.history.push({
                            event: "BatchTransfer", actor: batch.proposer, from: oldOwner, to: newOwner, batchId: batchId,
                            timestamp: commitBlock.timestamp, details: `Transferred via committed batch ${batchId}.`, txHash: commitReceipt.transactionHash
                        });
                        if (ctx.nodes && ctx.nodes[oldOwner]) {
                             if (!ctx.nodes[oldOwner].interactions) ctx.nodes[oldOwner].interactions = [];
                            ctx.nodes[oldOwner].interactions.push({ type: "TransferNFTViaBatch", tokenId: tokenIdStr, to: newOwner, batchId: batchId, timestamp: commitBlock.timestamp, txHash: commitReceipt.transactionHash });
                        }
                        if (ctx.nodes && ctx.nodes[newOwner]) {
                             if (!ctx.nodes[newOwner].interactions) ctx.nodes[newOwner].interactions = [];
                            ctx.nodes[newOwner].interactions.push({ type: "ReceiveNFTViaBatch", tokenId: tokenIdStr, from: oldOwner, batchId: batchId, timestamp: commitBlock.timestamp, txHash: commitReceipt.transactionHash });
                        }
                    }
                });
            }
            return ctx;
        });

        if (isCommitted) {
            console.log("    Exchange Batch successfully committed!");
            console.log("    Ownership changes verified via context update.");
        } else if (isFlagged) {
            console.log("    Batch was flagged, not committed.");
        } else {
            console.log("    Batch commit failed.");
        }

        // After commit batch, verify ownership and log results (giống script cũ)
        if (isCommitted) {
            console.log("\n    Verifying ownership changes after exchange batch commit:");
            // Giao dịch 1: Product 2 từ Retailer sang Buyer1
            let newOwnerP2 = await supplyChainNFT.ownerOf(tokenId2);
            console.log(`      Product ${product2.uniqueProductID} (Token ID: ${tokenId2}) new owner: ${newOwnerP2} (Expected: ${buyer1Acc.address})`);
            if (newOwnerP2.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${tokenId2}!`);
            }
            // Giao dịch 2: Product 3 từ Retailer sang Buyer1
            let newOwnerP3 = await supplyChainNFT.ownerOf(tokenId3);
            console.log(`      Product ${product3.uniqueProductID} (Token ID: ${tokenId3}) new owner: ${newOwnerP3} (Expected: ${buyer1Acc.address})`);
            if (newOwnerP3.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${tokenId3}!`);
            }
            // Giao dịch 3: Product 1 từ Buyer1 sang Retailer
            let newOwnerP1 = await supplyChainNFT.ownerOf(tokenId1);
            console.log(`      Product ${product1.uniqueProductID} (Token ID: ${tokenId1}) new owner: ${newOwnerP1} (Expected: ${retailerAcc.address})`);
            if (newOwnerP1.toLowerCase() !== retailerAcc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${tokenId1}!`);
            }
            console.log("\n    Batch processing for product exchange completed. This demonstrates efficient, atomic handling of multiple transfers.");
        }
    } else {
        console.log("  Skipping commit attempt as supermajority was not reached.");
        // Update context to reflect non-commitment
        currentContext = readAndUpdateContext(ctx => {
            const batch = ctx.batches[batchId];
            if (batch) {
                batch.status = "CommitSkipped (Insufficient Votes)";
                batch.lastUpdateTimestamp = Math.floor(Date.now() / 1000); // Approximate time
            }
            return ctx;
        });
    }

    console.log("\n--- 05: Batch Processing Scenario Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });

