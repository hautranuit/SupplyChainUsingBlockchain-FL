const fs =require("fs");
const path = require("path");

// Helper function to read demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run 03_scenario_marketplace_and_purchase.js (or later scripts) first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

// Add a counter for unique file names if running multiple disputes in one script execution
let disputeIdCounter = 1; 

// Helper function for delays (still used by the queue processor)
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
// REMOVED: const INFURA_DELAY_MS = 500; // 0.5 second delay

// --- Transaction Queue System ---
const transactionQueue = [];
let isProcessingQueue = false;
const MIN_INTERVAL_MS = 750; // Minimum interval between finishing one tx and starting the next from queue

async function processTransactionQueue() {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    while (transactionQueue.length > 0) {
        const task = transactionQueue.shift(); // Get the oldest task
        console.log(`Executing from queue: ${task.description}`);
        try {
            const result = await task.action(); // Execute the async action
            if (task.callback) {
                task.callback(null, result); // Pass result to callback
            }
        } catch (error) {
            console.error(`Error executing task "${task.description}":`, error.message);
            if (task.callback) {
                task.callback(error, null); // Pass error to callback
            }
            // Stop processing on error to allow inspection and prevent cascading failures
            isProcessingQueue = false; 
            throw error; 
        }
        if (transactionQueue.length > 0) { // If there are more tasks, wait
            await delay(MIN_INTERVAL_MS);
        }
    }
    isProcessingQueue = false;
}

function addToTransactionQueue(actionDescription, actionFunction) {
    return new Promise((resolve, reject) => {
        transactionQueue.push({
            description: actionDescription,
            action: actionFunction, // Should be an async function that returns a promise (e.g., the transaction receipt)
            callback: (error, result) => {
                if (error) reject(error);
                else resolve(result);
            }
        });
        if (!isProcessingQueue) {
            processTransactionQueue().catch(err => {
                // This catches errors from processTransactionQueue itself if it re-throws,
                // or if an unhandled promise rejection occurs within it.
                // Individual task errors are primarily handled by the promise returned by addToTransactionQueue.
                console.error("Critical error in queue processing supervisor:", err);
                // Ensure isProcessingQueue is reset if processTransactionQueue fails critically
                isProcessingQueue = false; 
            });
        }
    });
}
// --- End Transaction Queue System ---

async function main() {
    const hre = require("hardhat"); // Get Hardhat Runtime Environment
    const ethers = hre.ethers; // Explicitly get ethers from HRE

    console.log("--- Starting 06: Dispute Resolution Scenario ---");

    // Define gas options to be used for transactions
    // Adjust these values based on current Amoy testnet conditions if needed
    const gasOptions = { 
        maxPriorityFeePerGas: ethers.parseUnits('30', 'gwei'),
        maxFeePerGas: ethers.parseUnits('100', 'gwei') 
    };
    console.log("Using gas options:", gasOptions);

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    const productDetails = context.productDetails;

    if (!contractAddress || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and productDetails are present.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }

    const deployer = signers[0];
    const manufacturer = signers[1];
    const buyer1 = signers[6]; 
    const designatedArbitrator = signers[7];
    const arbitratorCandidate1 = signers[2];
    const arbitratorCandidate2 = signers[3];

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // MODIFIED: Target DEMO_PROD_003 for dispute
    const productInfo = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_003"); 
    if (!productInfo || !productInfo.tokenId || !productInfo.currentOwnerAddress) {
        // MODIFIED: Updated error message for DEMO_PROD_003
        console.error("Product 3 (DEMO_PROD_003) details or owner not found in context. Run previous scripts, including those that might assign ownership of Product 3.");
        process.exit(1);
    }
    const tokenIdToDispute = productInfo.tokenId;
    // MODIFIED: Use productInfo consistently
    const disputingPartySigner = signers.find(s => s.address.toLowerCase() === productInfo.currentOwnerAddress.toLowerCase());

    if (!disputingPartySigner) {
        // MODIFIED: Use productInfo consistently
        console.error(`Could not find signer for disputing party ${productInfo.currentOwnerAddress}`);
        process.exit(1);
    }

    // MODIFIED: Use productInfo consistently
    console.log(`\n--- Disputing Product ${productInfo.uniqueProductID} (Token ID: ${tokenIdToDispute}) ---`);
    console.log(`Disputing Party (Current Owner): ${disputingPartySigner.address}`);
    const disputeReason = "Product received damaged, quality not as expected.";
    
    const evidenceData = {
        timestamp: new Date().toISOString(),
        disputeReason: disputeReason,
        // MODIFIED: Use productInfo consistently
        productID: productInfo.uniqueProductID,
        tokenId: tokenIdToDispute.toString(),
        images: ["image_proof1.jpg", "video_proof.mp4"], // These are now just references for backend
        description: "Detailed description of the damage and discrepancy."
    };
    // REMOVED: const evidenceCID = await uploadToIPFS(evidenceData, `evidence_dispute_${disputeIdCounter}_${tokenIdToDispute}.json`);
    // console.log(`  Generated Evidence CID: ${evidenceCID}`);
    const evidenceDataString = JSON.stringify(evidenceData);
    console.log(`  Prepared Evidence Data (to be processed by backend): ${evidenceDataString.substring(0,100)}...`);
    disputeIdCounter++;

    // console.log(`  Opening dispute for Token ID ${tokenIdToDispute} by ${disputingPartySigner.address}...`);
    // MODIFIED: Pass evidenceDataString instead of evidenceCID
    // let tx = await supplyChainNFT.connect(disputingPartySigner).openDispute(tokenIdToDispute, disputeReason, evidenceDataString);
    // let receipt = await tx.wait(1);
    // console.log(`    Transaction to open dispute sent. Gas Used: ${receipt.gasUsed.toString()}`);
    // await delay(INFURA_DELAY_MS); // REMOVED delay

    let receipt = await addToTransactionQueue(
        `Opening dispute for Token ID ${tokenIdToDispute} by ${disputingPartySigner.address}`,
        async () => {
            const tx = await supplyChainNFT.connect(disputingPartySigner).openDispute(tokenIdToDispute, disputeReason, evidenceDataString);
            const awaitedReceipt = await tx.wait(1);
            console.log(`    Transaction to open dispute sent. Gas Used: ${awaitedReceipt.gasUsed.toString()}`);
            return awaitedReceipt;
        }
    );

    let openDisputeEvent = receipt.events?.find(event => event.event === "DisputeOpened");
    if (!openDisputeEvent) {
         // Fallback for environments where logs need explicit parsing (less common with ethers v5+ and wait())
        // Ensure 'logs' are available on receipt; Hardhat's receipt usually has them.
        const rawLogs = receipt.logs || [];
        for (const log of rawLogs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "DisputeOpened") {
                    openDisputeEvent = parsedLog;
                    break;
                }
            } catch (e) { /* Ignore if log is not from this contract's ABI */ }
        }
    }
    
    if (!openDisputeEvent) {
        console.error("    ERROR: DisputeOpened event not found in transaction receipt.");
        throw new Error("DisputeOpened event not found and could not determine disputeId.");
    }
    
    const disputeId = openDisputeEvent.args.disputeId;
    // MODIFIED: Add a check for openDisputeEvent.args.disputer before logging
    const disputerAddress = openDisputeEvent.args.disputer;
    console.log(`    DisputeOpened event processed. Dispute ID: ${disputeId.toString()}, Token ID: ${openDisputeEvent.args.tokenId.toString()}, Disputer: ${disputerAddress ? disputerAddress.toString() : 'undefined (from event)'}`);

    // console.log(`\\\\n  Proposing arbitrator candidates for Dispute ID ${disputeId.toString()}...`);
    // await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate1.address)).wait();
    // console.log(`    Proposed Candidate 1: ${arbitratorCandidate1.address}`);
    // await delay(INFURA_DELAY_MS); // REMOVED delay
    await addToTransactionQueue(
        `Proposing arbitrator candidate 1: ${arbitratorCandidate1.address} for Dispute ID ${disputeId.toString()}`,
        async () => {
            // MODIFIED: Added gasOptions
            const tx = await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate1.address, gasOptions);
            await tx.wait(1);
            console.log(`    Proposed Candidate 1: ${arbitratorCandidate1.address}`);
        }
    );

    // await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate2.address)).wait();
    // console.log(`    Proposed Candidate 2: ${arbitratorCandidate2.address}`);
    // await delay(INFURA_DELAY_MS); // REMOVED delay
    await addToTransactionQueue(
        `Proposing arbitrator candidate 2: ${arbitratorCandidate2.address} for Dispute ID ${disputeId.toString()}`,
        async () => {
            // MODIFIED: Added gasOptions
            const tx = await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate2.address, gasOptions);
            await tx.wait(1);
            console.log(`    Proposed Candidate 2: ${arbitratorCandidate2.address}`);
        }
    );

    // await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, designatedArbitrator.address)).wait();
    // console.log(`    Proposed Candidate 3 (Designated): ${designatedArbitrator.address}`);
    // await delay(INFURA_DELAY_MS); // REMOVED delay
    await addToTransactionQueue(
        `Proposing arbitrator candidate 3 (Designated): ${designatedArbitrator.address} for Dispute ID ${disputeId.toString()}`,
        async () => {
            // MODIFIED: Added gasOptions
            const tx = await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, designatedArbitrator.address, gasOptions);
            await tx.wait(1);
            console.log(`    Proposed Candidate 3 (Designated): ${designatedArbitrator.address}`);
        }
    );
    
    console.log("\\n  Voting for arbitrator candidates...");
    const voters = [buyer1, manufacturer, signers[2], signers[3]]; 
    for (const voter of voters) {
        // console.log(`    Voter ${voter.address} voting for ${designatedArbitrator.address}...`);
        try {
            // tx = await supplyChainNFT.connect(voter).voteForArbitrator(disputeId, designatedArbitrator.address);
            // receipt = await tx.wait(1);
            // console.log(`      Vote cast. Gas Used: ${receipt.gasUsed.toString()}`);
            // await delay(INFURA_DELAY_MS); // REMOVED delay
            await addToTransactionQueue(
                `Voter ${voter.address} voting for ${designatedArbitrator.address} in Dispute ID ${disputeId.toString()}`,
                async () => {
                    // MODIFIED: Added gasOptions
                    const tx = await supplyChainNFT.connect(voter).voteForArbitrator(disputeId, designatedArbitrator.address, gasOptions);
                    const voteReceipt = await tx.wait(1);
                    console.log(`      Vote cast by ${voter.address}. Gas Used: ${voteReceipt.gasUsed.toString()}`);
                    return voteReceipt;
                }
            );
        } catch (e) {
            console.warn(`      WARN: Voter ${voter.address} could not vote: ${e.message}`);
        }
    }

    // console.log(`\\\\n  Selecting arbitrator for Dispute ID ${disputeId.toString()}...`);
    // tx = await supplyChainNFT.connect(deployer).selectArbitrator(disputeId);
    // receipt = await tx.wait(1);
    // console.log(`    Transaction to select arbitrator sent. Gas Used: ${receipt.gasUsed.toString()}`);
    // await delay(INFURA_DELAY_MS); // REMOVED delay
    receipt = await addToTransactionQueue(
        `Selecting arbitrator for Dispute ID ${disputeId.toString()}`,
        async () => {
            // MODIFIED: Added gasOptions
            const tx = await supplyChainNFT.connect(deployer).selectArbitrator(disputeId, gasOptions);
            const awaitedReceipt = await tx.wait(1);
            console.log(`    Transaction to select arbitrator sent. Gas Used: ${awaitedReceipt.gasUsed.toString()}`);
            return awaitedReceipt;
        }
    );

    let arbitratorSelectedEvent = receipt.events?.find(event => event.event === "ArbitratorSelected");
    if (!arbitratorSelectedEvent) {
        const rawLogs = receipt.logs || [];
        for (const log of rawLogs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ArbitratorSelected") {
                    arbitratorSelectedEvent = parsedLog;
                    break;
                }
            } catch (e) { /* Ignore */ }
        }
    }

    let selectedArbitrator;
    if (!arbitratorSelectedEvent) {
        console.warn("    WARN: ArbitratorSelected event not found in receipt. Fetching from contract state...");
        const disputeData = await supplyChainNFT.disputesData(disputeId);
        if (disputeData.selectedArbitrator !== ethers.ZeroAddress) { // MODIFIED: ethers.constants.AddressZero to ethers.ZeroAddress
            selectedArbitrator = disputeData.selectedArbitrator;
            console.log(`    Fallback: Directly fetched selected arbitrator: ${selectedArbitrator}`);
            if (selectedArbitrator.toLowerCase() !== designatedArbitrator.address.toLowerCase()) {
                console.warn("    WARN (Fallback): Designated arbitrator was NOT selected. The demo might not proceed as planned.");
            }
        } else {
            throw new Error("ArbitratorSelected event not found and could not determine selected arbitrator from contract state.");
        }
    } else {
        selectedArbitrator = arbitratorSelectedEvent.args.selectedArbitrator;
        console.log(`    ArbitratorSelected event processed. Selected: ${selectedArbitrator}, Dispute ID: ${arbitratorSelectedEvent.args.disputeId.toString()}, Token ID: ${arbitratorSelectedEvent.args.tokenId.toString()}`);
        if (selectedArbitrator.toLowerCase() !== designatedArbitrator.address.toLowerCase()) {
            console.warn("    WARN: Designated arbitrator was NOT selected. The demo might not proceed as planned.");
        }
    }

    if (selectedArbitrator.toLowerCase() === designatedArbitrator.address.toLowerCase()) {
        console.log(`\n--- Dispute Resolution Process for Dispute ID ${disputeId.toString()} by Arbitrator ${designatedArbitrator.address} ---`);
        
        const resolutionDetailsText = "Arbitrator decision: Full refund to buyer, product to be returned to Manufacturer.";
        const resolutionOutcome = 1; // 1 = Favor Plaintiff (Buyer)

        const resolutionData = {
            timestamp: new Date().toISOString(),
            disputeId: disputeId.toString(),
            arbitrator: designatedArbitrator.address,
            decision: resolutionDetailsText,
            outcome: resolutionOutcome,
            actionsRequired: [
                // MODIFIED: Refer to disputingPartySigner
                `Disputing Party (${disputingPartySigner.address}) to have product (Token ID: ${tokenIdToDispute}) returned to Manufacturer (${manufacturer.address})`,
                // MODIFIED: Refer to disputingPartySigner
                `Manufacturer (${manufacturer.address}) to fund, and contract to issue, full refund to Disputing Party (${disputingPartySigner.address}).`
            ],
            settlementTerms: "NFT return and full refund to be enforced on-chain."
        };
        // REMOVED: const resolutionCID = await uploadToIPFS(resolutionData, `resolution_dispute_${disputeId.toString()}.json`);
        // console.log(`  Generated Resolution CID for IPFS: ${resolutionCID}`);
        const resolutionDataString = JSON.stringify(resolutionData);
        console.log(`  Prepared Resolution Data (to be processed by backend): ${resolutionDataString.substring(0,100)}...`);

        // 1. Record Decision
        // console.log(`\\\\n  1. Arbitrator (${designatedArbitrator.address}) recording decision for Dispute ID ${disputeId.toString()}...`);
        // MODIFIED: Pass resolutionDataString as the third argument (previously resolutionCID)
        // tx = await supplyChainNFT.connect(designatedArbitrator).recordDecision(disputeId, resolutionDetailsText, resolutionDataString, resolutionOutcome);
        // receipt = await tx.wait(1);
        // console.log(`    Transaction to record decision sent. Gas Used: ${receipt.gasUsed.toString()}`);
        // await delay(INFURA_DELAY_MS); // REMOVED delay
        receipt = await addToTransactionQueue(
            `Arbitrator (${designatedArbitrator.address}) recording decision for Dispute ID ${disputeId.toString()}`,
            async () => {
                const tx = await supplyChainNFT.connect(designatedArbitrator).recordDecision(disputeId, resolutionDetailsText, resolutionDataString, resolutionOutcome, gasOptions);
                const awaitedReceipt = await tx.wait(1);
                console.log(`    Transaction to record decision sent. Gas Used: ${awaitedReceipt.gasUsed.toString()}`);
                return awaitedReceipt;
            }
        );
        
        let decisionRecordedEvent = receipt.events?.find(e => e.event === "DisputeDecisionRecorded");
        if (!decisionRecordedEvent) {
            const rawLogs = receipt.logs || [];
            for (const log of rawLogs) {
                try {
                    const parsedLog = supplyChainNFT.interface.parseLog(log);
                    if (parsedLog && parsedLog.name === "DisputeDecisionRecorded") {
                        decisionRecordedEvent = parsedLog;
                        break;
                    }
                } catch (e) { /* Ignore */ }
            }
        }

        if (decisionRecordedEvent) {
            // MODIFIED: Use event.args.resolutionDetails to match contract event
            console.log(`      DisputeDecisionRecorded: DisputeID=${decisionRecordedEvent.args.disputeId}, Arbitrator=${decisionRecordedEvent.args.arbitrator}, Outcome=${decisionRecordedEvent.args.outcome}, ResolutionDetails=${decisionRecordedEvent.args.resolutionDetails}, ResolutionDataString (first 100 chars)=${(decisionRecordedEvent.args.resolutionDataString || '').substring(0,100)}...`);
        } else {
            console.warn("      WARN: Could not find DisputeDecisionRecorded event in receipt (or event signature mismatch).");
        }

        // 2. Enforce NFT Return
        const returnToAddress = manufacturer.address; 
        console.log(`\n  2. Arbitrator enforcing NFT (Token ID: ${tokenIdToDispute}) return to Manufacturer (${returnToAddress}) for Dispute ID ${disputeId.toString()}...`);
        
        const ownerBeforeReturn = await supplyChainNFT.ownerOf(tokenIdToDispute);
        console.log(`     Current owner of Token ID ${tokenIdToDispute}: ${ownerBeforeReturn}`);
        // MODIFIED: Check against disputingPartySigner.address instead of buyer1.address
        if (ownerBeforeReturn.toLowerCase() !== disputingPartySigner.address.toLowerCase()) {
            // This warning will now only appear if the owner fetched by ownerOf is not the disputing party.
            console.warn(`     WARN: Current owner ${ownerBeforeReturn} does not match the disputing party ${disputingPartySigner.address}. This might be unexpected. Proceeding with enforcement.`);
        }

        tx = await supplyChainNFT.connect(designatedArbitrator).enforceNFTReturn(disputeId, returnToAddress, gasOptions);
        receipt = await tx.wait(1);
        console.log(`    Transaction to enforce NFT return sent. Gas Used: ${receipt.gasUsed.toString()}`);
        // await delay(INFURA_DELAY_MS); // REMOVED delay

        let nftReturnEnforcedEvent = receipt.events?.find(e => e.event === "NFTReturnEnforced");
        if (!nftReturnEnforcedEvent) {
            const rawLogs = receipt.logs || [];
            for (const log of rawLogs) {
                try {
                    const parsedLog = supplyChainNFT.interface.parseLog(log);
                    if (parsedLog && parsedLog.name === "NFTReturnEnforced") {
                        nftReturnEnforcedEvent = parsedLog;
                        break;
                    }
                } catch (e) { /* Ignore */ }
            }
        }

        if (nftReturnEnforcedEvent) {
            console.log(`      NFTReturnEnforced: DisputeID=${nftReturnEnforcedEvent.args.disputeId}, TokenID=${nftReturnEnforcedEvent.args.tokenId}, From=${nftReturnEnforcedEvent.args.from}, To=${nftReturnEnforcedEvent.args.to}`);
        } else {
            console.warn("      WARN: Could not find NFTReturnEnforced event in receipt.");
        }

        const ownerAfterReturn = await supplyChainNFT.ownerOf(tokenIdToDispute);
        console.log(`     New owner of Token ID ${tokenIdToDispute}: ${ownerAfterReturn} (Expected: ${returnToAddress})`);
        if (ownerAfterReturn.toLowerCase() !== returnToAddress.toLowerCase()) {
            console.error(`       ERROR: NFT ownership verification failed after enforcement! Owner is ${ownerAfterReturn}.`);
        } else {
            console.log("       NFT ownership successfully transferred to Manufacturer.");
        }
        
        // 3. Enforce Refund
        let calculatedRefundAmountInPOL;

        // In this specific scenario (06_scenario_dispute_resolution.cjs):
        // - The dispute is initiated by Buyer1 for Product1.
        // - The arbitrator's decision (resolutionOutcome) is hardcoded to 1 (Favor Plaintiff/Buyer).
        // Therefore, the refund should be the original purchase price of Product1 plus a 0.01 POL compensation fee.
        // We must ensure productInfo and its purchasePrice (or price) are valid from demo_context.json.

        // Try 'purchasePrice' first, then fall back to 'price' for compatibility
        // MODIFIED: Use productInfo consistently
        let priceToUseString = productInfo ? (productInfo.purchasePrice || productInfo.price) : undefined;
        let originalPriceInWei;
        let originalPriceForLog; // For logging

        // MODIFIED: Use productInfo consistently
        if (productInfo && typeof priceToUseString === 'string' && priceToUseString.trim() !== "") {
            try {
                // priceToUseString is in GWEI as per user's information.
                // compensationFeePOL is 0.01 Ether.

                if (productInfo.uniqueProductID === "DEMO_PROD_002") {
                    console.log(`    INFO: For product ${productInfo.uniqueProductID}, using a fixed price of 0.2 POL (parsed as ETH) for refund calculation, overriding demo_context.json value ('${priceToUseString}' Gwei).`);
                    originalPriceInWei = ethers.parseUnits("0.2", "ether"); // 0.2 POL, assuming 18 decimals like ETH
                    originalPriceForLog = "0.2 POL";
                } else {
                    // priceToUseString from demo_context.json is a string representing the value in Wei.
                    // Example: "100000000000000000" for 0.1 POL/ETH.
                    // Convert this string directly to a BigInt to get the Wei value.
                    originalPriceInWei = BigInt(priceToUseString);
                    // For logging, format the Wei BigInt to a human-readable POL/ETH string.
                    originalPriceForLog = `${ethers.formatEther(originalPriceInWei)} POL`;
                    console.log(`    INFO: For product ${productInfo.uniqueProductID}, using price from demo_context.json: ${priceToUseString} Wei (which is ${originalPriceForLog}).`);
                }
                
                const compensationFeeInWei = ethers.parseEther("0.01"); // 0.01 ETH/POL in Wei

                refundAmount = originalPriceInWei + compensationFeeInWei; // MODIFIED: Changed .add() to + for bigint arithmetic

                // For logging:
                // const originalPriceInGweiForLog = priceToUseString; // The value from context file // No longer always GWEI
                const compensationFeeInEthForLog = "0.01";
                calculatedRefundAmountForLog = ethers.formatEther(refundAmount); // MODIFIED: ethers.utils.formatEther to ethers.formatEther // Total refund in Ether string

                console.log(`  Refund calculation for Disputing Party (${disputingPartySigner.address}) (dispute favored): Original Price (${originalPriceForLog}) + Compensation (${compensationFeeInEthForLog} ETH) = ${calculatedRefundAmountForLog} ETH.`);

            } catch (e) {
                // Catch errors from parseUnits, parseEther, or add.
                console.error(`  CRITICAL ERROR: Failed to calculate refund amount. Error: ${e.message}`);
                // Re-throw to halt further execution of the refund process, as the amount is undetermined.
                throw e;
            }
        } else {
            // This case should ideally not be reached if script 03 ran correctly and demo_context.json is populated with purchasePrice or price.
            // MODIFIED: Use productInfo consistently for logging
            const priceValForLog = productInfo ? (productInfo.purchasePrice !== undefined ? productInfo.purchasePrice : (productInfo.price !== undefined ? productInfo.price : "undefined field")) : "productInfo missing";
            // MODIFIED: Update default product ID in error message if productInfo is missing
            const prodId = productInfo ? productInfo.uniqueProductID : "DEMO_PROD_002 (productInfo missing)";
            const errorMessage = `  CRITICAL ERROR: Purchase price (checked 'purchasePrice' then 'price') for product ${prodId} is missing or invalid in demo_context.json (value found: '${priceValForLog}', expected Gwei string). Cannot determine refund amount.`;
            console.error(errorMessage);
            throw new Error(errorMessage.trim());
        }

        // refundAmount is now already calculated as a BigNumber in Wei.
        // The previous debug logs related to parseEther are removed.
        
        // MODIFIED: refundTo is the disputingPartySigner
        const refundTo = disputingPartySigner.address;
        const refundFrom = manufacturer.address; // Conceptual source, contract balance is used.

        // Log the amount to be deposited in ETH for readability
        console.log(`\n  Preparing for refund: Manufacturer (${manufacturer.address}) depositing ${ethers.formatEther(refundAmount)} ETH into the contract...`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther
        try {
            // Use refundAmount (BigNumber in Wei) directly in the transaction value
            // const depositTx = await supplyChainNFT.connect(manufacturer).depositDisputeFunds({ value: refundAmount });
            // const depositReceipt = await depositTx.wait(1);
            // console.log(`    Manufacturer deposited funds successfully. Gas Used: ${depositReceipt.gasUsed.toString()}`);
            // await delay(INFURA_DELAY_MS); // REMOVED delay
            const depositReceipt = await addToTransactionQueue(
                `Manufacturer (${manufacturer.address}) depositing ${ethers.formatEther(refundAmount)} ETH for refund`, // MODIFIED: ethers.utils.formatEther to ethers.formatEther
                async () => {
                    // MODIFIED: Added gasOptions
                    const tx = await supplyChainNFT.connect(manufacturer).depositDisputeFunds({ ...gasOptions, value: refundAmount });
                    const awaitedReceipt = await tx.wait(1);
                    console.log(`    Manufacturer deposited funds successfully. Gas Used: ${awaitedReceipt.gasUsed.toString()}`);
                    return awaitedReceipt;
                }
            );

            let fundsDepositedEvent = depositReceipt.events?.find(e => e.event === "FundsDeposited");
            if(fundsDepositedEvent) {
                console.log(`      FundsDeposited: Depositor=${fundsDepositedEvent.args.depositor}, Amount=${ethers.formatEther(fundsDepositedEvent.args.amount)} ETH`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther
            }

        } catch (e) {
            console.error(`    ERROR: Manufacturer failed to deposit funds: ${e.message}`);
            console.log("    Skipping refund enforcement due to deposit failure. Ensure contract can receive funds and manufacturer has ETH balance.");
        }

        const contractBalanceBeforeRefund = await ethers.provider.getBalance(contractAddress); // MODIFIED: Use contractAddress string directly
        console.log(`    Contract balance before attempting refund: ${ethers.formatEther(contractBalanceBeforeRefund)} ETH`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther

        if (contractBalanceBeforeRefund >= refundAmount) { // MODIFIED: Use >= for bigint comparison
            // MODIFIED: Log refers to disputingPartySigner
            // Log the refund amount in ETH for readability
            console.log(`\n  3. Arbitrator enforcing Refund of ${ethers.formatEther(refundAmount)} ETH to Disputing Party (${refundTo}) for Dispute ID ${disputeId.toString()}...`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther
            console.log(`     Refund will be sourced from contract balance (notionally from Manufacturer: ${refundFrom}).`);
            
            const buyerBalanceBeforeRefund = await ethers.provider.getBalance(refundTo);

            // Use refundAmount (BigNumber in Wei) directly in the transaction
            // MODIFIED: Corrected argument order for enforceRefund
            tx = await supplyChainNFT.connect(designatedArbitrator).enforceRefund(disputeId, refundTo, refundFrom, refundAmount, gasOptions);
            receipt = await tx.wait(1);
            console.log(`    Transaction to enforce refund sent. Gas Used: ${receipt.gasUsed.toString()}`);
            // await delay(INFURA_DELAY_MS); // REMOVED delay as INFURA_DELAY_MS is not defined and queue handles delays

            let refundEnforcedEvent = receipt.events?.find(e => e.event === "RefundEnforced");
            if (refundEnforcedEvent) {
                console.log(`      RefundEnforced: DisputeID=${refundEnforcedEvent.args.disputeId}, To=${refundEnforcedEvent.args.refundTo}, From=${refundEnforcedEvent.args.refundFrom}, Amount=${ethers.formatEther(refundEnforcedEvent.args.amount)} ETH`);
            } else {
                console.warn("      WARN: Could not find RefundEnforced event in receipt.");
            }

            const buyerBalanceAfterRefund = await ethers.provider.getBalance(refundTo);
            const contractBalanceAfterRefund = await ethers.provider.getBalance(contractAddress); // MODIFIED: Use contractAddress string directly
            // MODIFIED: Log refers to disputingPartySigner
            console.log(`     Disputing Party's balance after refund: ${ethers.formatEther(buyerBalanceAfterRefund)} ETH (Change: ${ethers.formatEther(buyerBalanceAfterRefund - buyerBalanceBeforeRefund)} ETH)`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther, .sub to -
        } else {
            // MODIFIED: Log refers to disputingPartySigner
            // Log the refund amount in ETH for readability
            console.error(`    ERROR: Contract balance (${ethers.formatEther(contractBalanceBeforeRefund)} ETH) is insufficient for refund amount (${ethers.formatEther(refundAmount)} ETH).`); // MODIFIED: ethers.utils.formatEther to ethers.formatEther
            console.log(`    Skipping refund enforcement for Disputing Party (${refundTo}).`);
        }

        // 4. Conclude Dispute
        // console.log(`\\\\n  4. Arbitrator concluding Dispute ID ${disputeId.toString()}...`);
        // tx = await supplyChainNFT.connect(designatedArbitrator).concludeDispute(disputeId);
        // receipt = await tx.wait(1);
        // console.log(`    Transaction to conclude dispute sent. Gas Used: ${receipt.gasUsed.toString()}`);
        // await delay(INFURA_DELAY_MS); // REMOVED delay
        receipt = await addToTransactionQueue(
            `Arbitrator concluding Dispute ID ${disputeId.toString()}`,
            async () => {
                // MODIFIED: Added gasOptions
                const tx = await supplyChainNFT.connect(designatedArbitrator).concludeDispute(disputeId, gasOptions);
                const awaitedReceipt = await tx.wait(1);
                console.log(`    Transaction to conclude dispute sent. Gas Used: ${awaitedReceipt.gasUsed.toString()}`);
                return awaitedReceipt;
            }
        );
        
        let disputeConcludedEvent = receipt.events?.find(e => e.event === "DisputeConcluded");
        if (!disputeConcludedEvent) {
            const rawLogs = receipt.logs || [];
            for (const log of rawLogs) {
                try {
                    const parsedLog = supplyChainNFT.interface.parseLog(log);
                    if (parsedLog && parsedLog.name === "DisputeConcluded") {
                        disputeConcludedEvent = parsedLog;
                        break;
                    }
                } catch (e) { /* Ignore */ }
            }
        }

        if (disputeConcludedEvent) {
            console.log(`      DisputeConcluded: DisputeID=${disputeConcludedEvent.args.disputeId}, WasEnforced=${disputeConcludedEvent.args.wasEnforced}, ConcludedBy=${disputeConcludedEvent.args.concludedBy}, Timestamp=${new Date(disputeConcludedEvent.args.timestamp.toNumber() * 1000).toISOString()}`);
        } else {
            console.warn("      WARN: Could not find DisputeConcluded event in receipt.");
        }

        console.log(`\n--- Dispute ID ${disputeId.toString()} fully processed and concluded. ---`);

    } else {
        console.log(`\n  Skipping dispute resolution steps as the designated arbitrator (${designatedArbitrator.address}) was not selected. Actual selected: ${selectedArbitrator}`);
    }
    console.log("\n--- 06: Dispute Resolution Scenario Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

