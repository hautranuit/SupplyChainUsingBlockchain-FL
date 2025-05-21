const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run 04_scenario_transport_and_ipfs.js first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

async function main() {
    console.log("--- Starting 05: Batch Processing Scenarios - Product Exchange ---");
    const metrics = {}; // Initialize metrics object

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    const productDetails = context.productDetails;

    if (!contractAddress || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and productDetails are present.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    // Signers configured in script 01:
    // deployer = signers[0]; manufacturerAcc = signers[1];
    // transporter1Acc = signers[2]; transporter2Acc = signers[3]; 
    // transporter3Acc = signers[4]; // Previously batchProposerAcc
    const retailerAcc = signers[5]; 
    const buyer1Acc = signers[6]; // Buyer1 will be the batch proposer (Secondary Node)
    // arbitratorAcc = signers[7];
    
    const batchProposerAcc = buyer1Acc; // Buyer1 proposes the exchange batch

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }
    const deployer = signers[0];
    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract SupplyChainNFT.");

    // --- Kịch bản trao đổi sản phẩm qua Batch ---
    // Buyer1 (batchProposerAcc) muốn đổi Product 1 của mình lấy Product 2 và Product 3 từ Retailer (retailerAcc).

    console.log("\\n--- Product Exchange Simulation via Batch ---");
    console.log(`Proposer (Buyer1 - Secondary Node): ${batchProposerAcc.address}`);
    console.log(`Counterparty (Retailer): ${retailerAcc.address}`);

    // Lấy thông tin sản phẩm từ context (giả định script 04 đã cập nhật đúng)
    // Product 1: DEMO_PROD_001, thuộc sở hữu của Buyer1
    // Product 2: DEMO_PROD_002, thuộc sở hữu của Retailer
    // Product 3: DEMO_PROD_003, thuộc sở hữu của Retailer (cần thêm vào demo_context nếu chưa có)

    const product1Context = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_001");
    const product2Context = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_002");
    const product3Context = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_003"); // Giả sử Product 3 được tạo và chuyển cho Retailer ở script trước

    if (!product1Context || !product2Context || !product3Context) {
        console.error("Could not find required product details (DEMO_PROD_001, DEMO_PROD_002, DEMO_PROD_003) in demo_context.json. Ensure previous scripts ran successfully and updated context.");
        if (!product1Context) console.error("Missing: DEMO_PROD_001");
        if (!product2Context) console.error("Missing: DEMO_PROD_002");
        if (!product3Context) console.error("Missing: DEMO_PROD_003");
        process.exit(1);
    }
    
    // Kiểm tra tokenId và currentOwnerAddress
    if (!product1Context.tokenId || !product1Context.currentOwnerAddress ||
        !product2Context.tokenId || !product2Context.currentOwnerAddress ||
        !product3Context.tokenId || !product3Context.currentOwnerAddress) {
        console.error("Missing tokenId or currentOwnerAddress for one or more products in demo_context.json.");
        console.log("Product 1 Context:", product1Context);
        console.log("Product 2 Context:", product2Context);
        console.log("Product 3 Context:", product3Context);
        process.exit(1);
    }

    // Xác minh chủ sở hữu ban đầu (quan trọng cho logic script)
    if (product1Context.currentOwnerAddress.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
        console.error(`Error: Product 1 (Token ID: ${product1Context.tokenId}) is expected to be owned by Buyer1 (${buyer1Acc.address}), but found owner: ${product1Context.currentOwnerAddress}. Check demo_context.json.`);
        process.exit(1);
    }
    if (product2Context.currentOwnerAddress.toLowerCase() !== retailerAcc.address.toLowerCase()) {
        console.error(`Error: Product 2 (Token ID: ${product2Context.tokenId}) is expected to be owned by Retailer (${retailerAcc.address}), but found owner: ${product2Context.currentOwnerAddress}. Check demo_context.json.`);
        process.exit(1);
    }
    if (product3Context.currentOwnerAddress.toLowerCase() !== retailerAcc.address.toLowerCase()) {
        console.error(`Error: Product 3 (Token ID: ${product3Context.tokenId}) is expected to be owned by Retailer (${retailerAcc.address}), but found owner: ${product3Context.currentOwnerAddress}. Check demo_context.json.`);
        process.exit(1);
    }

    console.log("Debug ethers.BigNumber:", ethers.BigNumber); // This will likely show undefined
    console.log("Debug ethers.utils:", ethers.utils); // In v5, BigNumber might have been here
    console.log("Using ethers.BigInt for conversion if available, or global BigInt");

    const transactionsForExchangeBatch = [
        { // Buyer1 chuyển Product 1 cho Retailer
            from: buyer1Acc.address,
            to: retailerAcc.address,
            tokenId: BigInt(product1Context.tokenId) // Corrected: Use global BigInt
        },
        { // Retailer chuyển Product 2 cho Buyer1
            from: retailerAcc.address,
            to: buyer1Acc.address,
            tokenId: BigInt(product2Context.tokenId) // Corrected: Use global BigInt
        },
        { // Retailer chuyển Product 3 cho Buyer1
            from: retailerAcc.address,
            to: buyer1Acc.address,
            tokenId: BigInt(product3Context.tokenId) // Corrected: Use global BigInt
        }
    ];

    if (transactionsForExchangeBatch.length === 0) {
        console.log("No transactions to batch process. Exiting.");
        return;
    }

    console.log(`\\n--- Scenario: Proposing and Committing Product Exchange Batch (${transactionsForExchangeBatch.length} Transactions) ---`);
    console.log("Detailed transactions for the exchange batch:");
    console.log(`  1. Product ${product2Context.uniqueProductID} (Token ID: ${product2Context.tokenId}) from Retailer (${retailerAcc.address}) to Buyer1 (${buyer1Acc.address})`);
    console.log(`  2. Product ${product3Context.uniqueProductID} (Token ID: ${product3Context.tokenId}) from Retailer (${retailerAcc.address}) to Buyer1 (${buyer1Acc.address})`);
    console.log(`  3. Product ${product1Context.uniqueProductID} (Token ID: ${product1Context.tokenId}) from Buyer1 (${buyer1Acc.address}) to Retailer (${retailerAcc.address})`);
    
    console.log("\\nRationale for Batching these transactions:");
    console.log("  Instead of executing 3 separate NFT transfer transactions (each incurring gas fees and requiring individual confirmation), we are batching them.");
    console.log("  This approach offers several advantages:");
    console.log("    - Optimized Gas Costs: A single 'proposeBatch' and 'commitBatch' operation (plus validator votes) is generally more gas-efficient than multiple individual 'transferFrom' calls, especially as the number of batched items grows.");
    console.log("    - Improved Time Efficiency: Waiting for confirmations for fewer batch-level transactions is faster than for many individual ones.");
    console.log("    - Enhanced Atomicity: The entire exchange (all 3 transfers) either succeeds together or fails together if the batch isn't committed. This prevents partial exchanges, which is crucial for such multi-party swaps.");

    console.log(`\\nBatch Proposer (Buyer1): ${batchProposerAcc.address}`);

    // Kiểm tra quyền của Proposer (Buyer1) để chuyển Product 1
    // và quyền của Retailer để chuyển Product 2 & 3 (thông qua việc Proposer tạo batch)
    // Smart contract sẽ kiểm tra quyền sở hữu và phê duyệt (approval) khi commit batch.

    let tx = await supplyChainNFT.connect(batchProposerAcc).proposeBatch(transactionsForExchangeBatch);
    let receipt = await tx.wait(1);
    console.log(`  Batch proposed for exchange. Gas Used: ${receipt.gasUsed.toString()}`);

    let proposeEvent;
    for (const log of receipt.logs) {
        try {
            const parsedLog = supplyChainNFT.interface.parseLog(log);
            if (parsedLog && parsedLog.name === "BatchProposed") {
                proposeEvent = parsedLog;
                break;
            }
        } catch (e) { /* Ignore */ }
    }
    if (!proposeEvent) throw new Error("BatchProposed event not found.");
    const batchId = proposeEvent.args.batchId;
    const selectedValidators = proposeEvent.args.selectedValidators; // These are Primary Nodes
    console.log(`  Batch ID for Exchange: ${batchId.toString()}, Selected Validators (${selectedValidators.length}): ${selectedValidators.join(", ")}`);

    console.log("\\n  Validators voting for the exchange batch (aiming for supermajority approval)...");
    let approvals = 0;
    const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
    const numSelectedValidators = selectedValidators.length;
    const requiredApprovals = Math.ceil(numSelectedValidators * Number(superMajorityFraction) / 100);
    console.log(`    Supermajority: ${superMajorityFraction.toString()}%, Num Selected: ${numSelectedValidators}, Required Approvals: ${requiredApprovals}`);

    for (let i = 0; i < numSelectedValidators; i++) {
        const validatorAddr = selectedValidators[i];
        const validatorSigner = signers.find(s => s.address === validatorAddr);
        if (!validatorSigner) {
            console.warn(`    Could not find signer for validator ${validatorAddr}. Skipping vote.`);
            continue;
        }
        // Let most validators approve
        const vote = approvals < requiredApprovals; 
        if (vote) approvals++;
        
        console.log(`    Validator ${validatorSigner.address} voting ${vote ? 'Approve' : 'Deny'}...`);
        tx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote);
        receipt = await tx.wait(1);
        console.log(`      Vote cast. Gas Used: ${receipt.gasUsed.toString()}`);
    }
    console.log(`    Total approvals: ${approvals} (Required: ${requiredApprovals})`);

    if (approvals >= requiredApprovals) {
        console.log("  Attempting to commit batch...");
        tx = await supplyChainNFT.connect(batchProposerAcc).commitBatch(batchId);
        receipt = await tx.wait(1);
        console.log(`    Batch commit attempted. Gas Used: ${receipt.gasUsed.toString()}`);

        const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
        if (batchDetails.committed) {
            console.log("    Exchange Batch successfully committed!");
            metrics.successfulBatches = (metrics.successfulBatches || 0) + 1;
            
            console.log("\\n    Verifying ownership changes after exchange batch commit:");
            // Giao dịch 1: Product 2 từ Retailer sang Buyer1
            let newOwnerP2 = await supplyChainNFT.ownerOf(product2Context.tokenId);
            console.log(`      Product ${product2Context.uniqueProductID} (Token ID: ${product2Context.tokenId}) new owner: ${newOwnerP2} (Expected: ${buyer1Acc.address})`);
            if (newOwnerP2.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${product2Context.tokenId}!`);
            }

            // Giao dịch 2: Product 3 từ Retailer sang Buyer1
            let newOwnerP3 = await supplyChainNFT.ownerOf(product3Context.tokenId);
            console.log(`      Product ${product3Context.uniqueProductID} (Token ID: ${product3Context.tokenId}) new owner: ${newOwnerP3} (Expected: ${buyer1Acc.address})`);
            if (newOwnerP3.toLowerCase() !== buyer1Acc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${product3Context.tokenId}!`);
            }

            // Giao dịch 3: Product 1 từ Buyer1 sang Retailer
            let newOwnerP1 = await supplyChainNFT.ownerOf(product1Context.tokenId);
            console.log(`      Product ${product1Context.uniqueProductID} (Token ID: ${product1Context.tokenId}) new owner: ${newOwnerP1} (Expected: ${retailerAcc.address})`);
            if (newOwnerP1.toLowerCase() !== retailerAcc.address.toLowerCase()) {
                console.error(`        ERROR: Ownership mismatch for Token ID ${product1Context.tokenId}!`);
            }
            console.log("\\n    Batch processing for product exchange completed. This demonstrates efficient, atomic handling of multiple transfers.");

            // --- Cập nhật demo_context.json ---
            console.log("\\n    Updating demo_context.json with new ownership and status...");
            const product1Index = context.productDetails.findIndex(p => p.uniqueProductID === "DEMO_PROD_001");
            const product2Index = context.productDetails.findIndex(p => p.uniqueProductID === "DEMO_PROD_002");
            const product3Index = context.productDetails.findIndex(p => p.uniqueProductID === "DEMO_PROD_003");

            if (product1Index !== -1) {
                context.productDetails[product1Index].currentOwnerAddress = retailerAcc.address;
                context.productDetails[product1Index].productStatus = "ExchangedInBatch_ToRetailer";
                console.log(`      Updated Product 1 (DEMO_PROD_001) owner to ${retailerAcc.address}, status to ExchangedInBatch_ToRetailer`);
            }
            if (product2Index !== -1) {
                context.productDetails[product2Index].currentOwnerAddress = buyer1Acc.address;
                context.productDetails[product2Index].productStatus = "ExchangedInBatch_ToBuyer1";
                console.log(`      Updated Product 2 (DEMO_PROD_002) owner to ${buyer1Acc.address}, status to ExchangedInBatch_ToBuyer1`);
            }
            if (product3Index !== -1) {
                context.productDetails[product3Index].currentOwnerAddress = buyer1Acc.address;
                context.productDetails[product3Index].productStatus = "ExchangedInBatch_ToBuyer1";
                console.log(`      Updated Product 3 (DEMO_PROD_003) owner to ${buyer1Acc.address}, status to ExchangedInBatch_ToBuyer1`);
            }
            
            const demoContextPath = path.join(__dirname, "demo_context.json");
            fs.writeFileSync(demoContextPath, JSON.stringify(context, null, 2));
            console.log(`    demo_context.json updated successfully at ${demoContextPath}`);
            // --- Kết thúc cập nhật demo_context.json ---

        } else if (batchDetails.flagged) {
            console.log("    Batch was flagged, not committed (as expected if approvals were insufficient or other issue).");
            metrics.failedBatches = (metrics.failedBatches || 0) + 1;
        } else {
            console.log("    Batch not committed and not flagged. Status unclear from direct fields.");
        }
    } else {
        console.log("  Skipping commit attempt as supermajority was not reached by this script's voting logic.");
        // To demonstrate a flagged batch, one would ensure denials lead to flagging.
        // This scenario focuses on a successful commit.
    }
    
    // TODO: Add a scenario for a flagged batch if required by the demo, similar to supplyChainConsensus.test.cjs

    // Update context (if necessary, e.g., new owners)
    // For now, just logging completion.
    fs.appendFileSync(path.join(__dirname, "demo_run.log"), `Product Exchange Batch Processing for Batch ID ${batchId ? batchId.toString() : 'N/A'} completed at ${new Date().toISOString()}\\n`);

    console.log("\\n--- 05: Batch Processing Scenarios (Product Exchange) Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

