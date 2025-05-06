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
    console.log("--- Starting 05: Batch Processing Scenarios ---");

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
    const batchProposerAcc = signers[4]; // transporter3Acc (Secondary Node)
    // retailerAcc = signers[5]; buyer1Acc = signers[6]; arbitratorAcc = signers[7];
    // Primary nodes (transporter1Acc, transporter2Acc) will act as validators.
    // For more validators, we would need more signers or re-use, but script 01 configured 5 primary nodes if we include deployer and others not explicitly named transporterXAcc.
    // For simplicity, we will use the configured primary nodes (T1, T2) and potentially others if the contract selects them.

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }
    const deployer = signers[0];
    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // Identify products that completed transport and might need a final batch transfer (e.g., to a final holding address or confirming state)
    // For this demo, let's assume the transport completion in script 04 means the buyer (current owner) now wants to formally log this via a batch process, 
    // or perhaps transfer to a different internal wallet. We will simulate a batch of transfers for products that completed transport.

    const product1Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_001" && p.transportStatus === "Completed");
    const product2Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_002" && p.transportStatus === "Completed");

    if (!product1Info || !product2Info) {
        console.error("Could not find completed transport details for Product 1 or Product 2. Run previous scripts.");
        process.exit(1);
    }

    const transactionsForBatch1 = [];
    // Example: Transfer Product 1 from current owner (buyer1) to a new internal wallet (e.g., retailerAcc for demo)
    if (product1Info.currentOwnerAddress) {
        transactionsForBatch1.push({
            from: product1Info.currentOwnerAddress,
            to: signers[5].address, // retailerAcc, let's say this is the final destination wallet
            tokenId: ethers.BigNumber.from(product1Info.tokenId) // Ensure tokenId is BigNumber
        });
    }
    // Example: Transfer Product 2 from current owner (buyer2/retailer) to another internal wallet (e.g., deployer for demo)
    if (product2Info.currentOwnerAddress) {
        transactionsForBatch1.push({
            from: product2Info.currentOwnerAddress,
            to: deployer.address, // deployer's address
            tokenId: ethers.BigNumber.from(product2Info.tokenId) // Ensure tokenId is BigNumber
        });
    }

    if (transactionsForBatch1.length === 0) {
        console.log("No transactions to batch process. Exiting.");
        return;
    }

    console.log(`\n--- Scenario 1: Proposing and Committing a Batch of ${transactionsForBatch1.length} Transactions ---`);
    console.log("Transactions to be batched:", transactionsForBatch1.map(t => ({from: t.from, to: t.to, tokenId: t.tokenId.toString() })));
    console.log(`Batch Proposer: ${batchProposerAcc.address}`);

    let tx = await supplyChainNFT.connect(batchProposerAcc).proposeBatch(transactionsForBatch1);
    let receipt = await tx.wait(1);
    console.log(`  Batch proposed. Gas Used: ${receipt.gasUsed.toString()}`);

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
    const selectedValidators = proposeEvent.args.selectedValidators;
    console.log(`  Batch ID: ${batchId.toString()}, Selected Validators (${selectedValidators.length}): ${selectedValidators.join(", ")}`);

    console.log("  Validators voting (aiming for supermajority approval)...`);
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
            console.log("    Batch successfully committed!");
            metrics.successfulBatches = (metrics.successfulBatches || 0) + 1;
            // Verify ownership changes
            for (const batchedTx of transactionsForBatch1) {
                const newOwner = await supplyChainNFT.ownerOf(batchedTx.tokenId);
                console.log(`      Token ID ${batchedTx.tokenId.toString()} new owner: ${newOwner} (Expected: ${batchedTx.to})`);
                if (newOwner !== batchedTx.to) {
                    console.error(`        ERROR: Ownership mismatch for Token ID ${batchedTx.tokenId.toString()}!`);
                }
            }
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
    fs.appendFileSync(path.join(__dirname, "demo_run.log"), `Batch Processing for Batch ID ${batchId.toString()} completed at ${new Date().toISOString()}\n`);

    console.log("--- 05: Batch Processing Scenarios Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

