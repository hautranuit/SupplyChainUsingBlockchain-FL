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

// Helper function for delays
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
// --- End Helper Function ---

async function main() {
    const ethers = hre.ethers; // Explicitly get ethers from HRE

    console.log("--- Starting 06: Dispute Resolution Scenario (Fixed Token ID Lookup) ---");

    // Static gas options
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'),
        maxFeePerGas: ethers.parseUnits('40', 'gwei')
    };

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
    const manufacturer = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Manufacturer"));
    const retailer = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Retailer"));
    const designatedArbitrator = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Arbitrator"));
    const arbitratorCandidate1 = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Transporter 1"));
    const arbitratorCandidate2 = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Transporter 2"));
    const buyer1 = signers.find(s => s.address.toLowerCase() === Object.keys(nodes).find(k => nodes[k].name === "Buyer/Customer")); // Should own Product 2 after exchange in script 05

    if (!deployer || !manufacturer || !retailer || !designatedArbitrator || !arbitratorCandidate1 || !arbitratorCandidate2 || !buyer1) {
        console.error("Error: Could not find one or more required accounts based on context.");
        process.exit(1);
    }

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // --- Find Product and Token ID Dynamically ---
    const uniqueIdToDispute = "DEMO_PROD_001";
    console.log(`\nLooking up Token ID for product with uniqueProductID: ${uniqueIdToDispute}...`);
    const productToDispute = Object.values(products).find(p => p.uniqueProductID === uniqueIdToDispute);

    if (!productToDispute || !productToDispute.tokenId) {
        console.error(`Error: Product with uniqueID ${uniqueIdToDispute} or its tokenId not found in context. Run previous scripts.`);
        process.exit(1);
    }
    const tokenIdToDispute = productToDispute.tokenId;
    console.log(`  Found ${uniqueIdToDispute} -> Token ID: ${tokenIdToDispute}`);

    // Disputing party is Retailer (current owner of Product01)
    const disputingPartySigner = retailer;
    console.log(`Disputing Party (Retailer): ${disputingPartySigner.address}`);

    // Verify ownership from context
    if (productToDispute.currentOwnerAddress.toLowerCase() !== disputingPartySigner.address.toLowerCase()) {
         console.error(`Error: Owner mismatch for Product ${uniqueIdToDispute} (Token ID: ${tokenIdToDispute}). Expected ${disputingPartySigner.address} (Retailer), found ${productToDispute.currentOwnerAddress}. Check context after script 05.`);
         process.exit(1);
    }

    console.log(`\n--- Disputing Product ${productToDispute.uniqueProductID} (Token ID: ${tokenIdToDispute}) ---`);
    const disputeReason = "Product received in damaged condition during batch exchange.";
    const evidenceData = {
        timestamp: new Date().toISOString(),
        disputeReason: disputeReason,
        productID: productToDispute.uniqueProductID,
        tokenId: tokenIdToDispute.toString(),
        images: ["damage_proof1.jpg", "package_condition.jpg"],
        description: "Product arrived with visible damage to packaging and internal components during batch exchange process."
    };
    const evidenceDataString = JSON.stringify(evidenceData);
    console.log(`  Prepared Evidence Data: ${evidenceDataString.substring(0, 100)}...`);

    // 1. Open Dispute
    console.log(`\n1. Opening dispute for Token ID ${tokenIdToDispute} by ${disputingPartySigner.address}...`);
    let openTx, openReceipt, openBlock, openDisputeEvent, disputeId;
    try {
        openTx = await supplyChainNFT.connect(disputingPartySigner).openDispute(tokenIdToDispute, disputeReason, evidenceDataString, gasOptions);
        openReceipt = await openTx.wait(1);
        openBlock = await ethers.provider.getBlock(openReceipt.blockNumber);
        console.log(`    Dispute opened. Gas Used: ${openReceipt.gasUsed.toString()}`);

        openDisputeEvent = openReceipt.logs?.map(log => {
            try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
        }).find(event => event?.name === "DisputeOpened");

        if (!openDisputeEvent) throw new Error("DisputeOpened event not found.");
        disputeId = openDisputeEvent.args.disputeId.toString();
        console.log(`    Dispute ID: ${disputeId}`);
    } catch (error) {
        console.error("  ERROR opening dispute:", error);
        process.exit(1);
    }

    // Update context after opening dispute
    currentContext = readAndUpdateContext(ctx => {
        if (!ctx.disputes) ctx.disputes = {};
        ctx.disputes[disputeId] = {
            disputeId: disputeId,
            tokenId: tokenIdToDispute,
            disputer: disputingPartySigner.address.toLowerCase(),
            reason: disputeReason,
            evidence: evidenceDataString,
            evidenceCID: null,
            status: "Opened",
            proposedCandidates: [],
            votes: {},
            selectedArbitrator: null,
            resolutionDetails: null,
            resolutionDataString: null,
            resolutionCID: null,
            resolutionOutcome: null,
            nftReturnEnforced: false,
            refundEnforced: false,
            enforced: false,
            openTimestamp: openBlock.timestamp,
            openTxHash: openReceipt.transactionHash,
            decisionTimestamp: null,
            enforcedTimestamp: null,
            lastUpdateTimestamp: openBlock.timestamp
        };
        if (ctx.products && ctx.products[tokenIdToDispute]) {
            ctx.products[tokenIdToDispute].disputeInfo = { disputeId: disputeId, status: "Opened" };
            ctx.products[tokenIdToDispute].lastUpdateTimestamp = openBlock.timestamp;
        }
        if (ctx.nodes && ctx.nodes[disputingPartySigner.address.toLowerCase()]) {
            if (!ctx.nodes[disputingPartySigner.address.toLowerCase()].interactions) ctx.nodes[disputingPartySigner.address.toLowerCase()].interactions = [];
            ctx.nodes[disputingPartySigner.address.toLowerCase()].interactions.push({
                type: "OpenDispute", disputeId: disputeId, tokenId: tokenIdToDispute, timestamp: openBlock.timestamp,
                details: `Opened dispute for product ${tokenIdToDispute}. Reason: ${disputeReason.substring(0, 50)}...`, txHash: openReceipt.transactionHash
            });
        }
        return ctx;
    });

    // 2. Propose Arbitrator Candidates
    console.log(`\n2. Proposing arbitrator candidates for Dispute ID ${disputeId}...`);
    const candidates = [arbitratorCandidate1, arbitratorCandidate2, designatedArbitrator];
    const candidateReceipts = {};
    try {
        for (const candidate of candidates) {
            console.log(`    Proposing Candidate: ${candidate.address}`);
            let proposeTx = await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, candidate.address, gasOptions);
            let proposeReceipt = await proposeTx.wait(1);
            let proposeBlock = await ethers.provider.getBlock(proposeReceipt.blockNumber);
            console.log(`      Proposed. Gas Used: ${proposeReceipt.gasUsed.toString()}`);
            candidateReceipts[candidate.address.toLowerCase()] = { receipt: proposeReceipt, block: proposeBlock };
            await delay(200); // Small delay between proposals
        }
    } catch (error) {
        console.error("  ERROR proposing candidates:", error);
        // Continue to context update, but state might be incomplete
    }

    // Update context with proposed candidates
    currentContext = readAndUpdateContext(ctx => {
        const dispute = ctx.disputes[disputeId];
        if (!dispute) return ctx;
        let maxTimestamp = dispute.lastUpdateTimestamp;
        for (const candidateAddr in candidateReceipts) {
            dispute.proposedCandidates.push({
                address: candidateAddr,
                proposeTimestamp: candidateReceipts[candidateAddr].block.timestamp,
                proposeTxHash: candidateReceipts[candidateAddr].receipt.transactionHash
            });
            maxTimestamp = Math.max(maxTimestamp, candidateReceipts[candidateAddr].block.timestamp);
            if (ctx.nodes && ctx.nodes[deployer.address.toLowerCase()]) {
                if (!ctx.nodes[deployer.address.toLowerCase()].interactions) ctx.nodes[deployer.address.toLowerCase()].interactions = [];
                 ctx.nodes[deployer.address.toLowerCase()].interactions.push({
                    type: "ProposeArbitratorCandidate", disputeId: disputeId, candidate: candidateAddr,
                    timestamp: candidateReceipts[candidateAddr].block.timestamp, txHash: candidateReceipts[candidateAddr].receipt.transactionHash
                 });
            }
        }
        dispute.status = "ProposingCandidates";
        dispute.lastUpdateTimestamp = maxTimestamp;
        return ctx;
    });

    // 3. Vote for Arbitrator Candidates
    console.log("\n3. Voting for arbitrator candidates...");
    
    // Define voters: Only signers[0] to signer[7], excluding the arbitrator candidates themselves
    const candidateAddresses = [arbitratorCandidate1.address.toLowerCase(), arbitratorCandidate2.address.toLowerCase(), designatedArbitrator.address.toLowerCase()];
    const availableVoters = signers.slice(0, 8).filter(signer => 
        !candidateAddresses.includes(signer.address.toLowerCase())
    );
    
    console.log(`    Available voters (excluding candidates): ${availableVoters.length} accounts`);
    
    // Create realistic voting distribution instead of everyone voting for the same candidate
    const voteDistribution = {};
    const voteReceipts = {};
    
    try {
        for (let i = 0; i < availableVoters.length; i++) {
            const voter = availableVoters[i];
            let voteTarget;
            
            // Create realistic voting pattern:
            // - First 40% vote for designatedArbitrator
            // - Next 30% vote for arbitratorCandidate1  
            // - Remaining 30% vote for arbitratorCandidate2
            const voterIndex = i / availableVoters.length;
            if (voterIndex < 0.4) {
                voteTarget = designatedArbitrator.address;
            } else if (voterIndex < 0.7) {
                voteTarget = arbitratorCandidate1.address;
            } else {
                voteTarget = arbitratorCandidate2.address;
            }
            
            console.log(`    Voter ${voter.address} (${nodes[voter.address.toLowerCase()]?.name || 'Unknown'}) voting for ${voteTarget}...`);
            
            try {
                let voteTx = await supplyChainNFT.connect(voter).voteForArbitrator(disputeId, voteTarget, gasOptions);
                let voteReceipt = await voteTx.wait(1);
                let voteBlock = await ethers.provider.getBlock(voteReceipt.blockNumber);
                console.log(`      Vote cast. Gas Used: ${voteReceipt.gasUsed.toString()}`);
                
                voteReceipts[voter.address.toLowerCase()] = { 
                    receipt: voteReceipt, 
                    block: voteBlock, 
                    voteTarget: voteTarget 
                };
                
                // Track vote distribution for logging
                if (!voteDistribution[voteTarget.toLowerCase()]) {
                    voteDistribution[voteTarget.toLowerCase()] = 0;
                }
                voteDistribution[voteTarget.toLowerCase()]++;
                
            } catch (e) {
                console.warn(`      WARN: Voter ${voter.address} could not vote: ${e.message}`);
            }
            await delay(200);
        }
        
        // Log vote distribution
        console.log("\n    Vote Distribution Summary:");
        for (const [candidateAddr, voteCount] of Object.entries(voteDistribution)) {
            const candidateName = candidateAddr === designatedArbitrator.address.toLowerCase() ? "Designated Arbitrator" :
                                 candidateAddr === arbitratorCandidate1.address.toLowerCase() ? "Arbitrator Candidate 1" :
                                 candidateAddr === arbitratorCandidate2.address.toLowerCase() ? "Arbitrator Candidate 2" : "Unknown";
            console.log(`      ${candidateName} (${candidateAddr}): ${voteCount} votes`);
        }
        
    } catch (error) {
        console.error("  ERROR during voting process:", error);
    }

    // Update context with votes
    currentContext = readAndUpdateContext(ctx => {
        const dispute = ctx.disputes[disputeId];
        if (!dispute) return ctx;
        let maxTimestamp = dispute.lastUpdateTimestamp;
        for (const voterAddr in voteReceipts) {
            const voteData = voteReceipts[voterAddr];
            if (!dispute.votes[voteData.voteTarget.toLowerCase()]) {
                dispute.votes[voteData.voteTarget.toLowerCase()] = [];
            }
            dispute.votes[voteData.voteTarget.toLowerCase()].push({
                voter: voterAddr,
                timestamp: voteData.block.timestamp,
                txHash: voteData.receipt.transactionHash
            });
            maxTimestamp = Math.max(maxTimestamp, voteData.block.timestamp);
            if (ctx.nodes && ctx.nodes[voterAddr]) {
                if (!ctx.nodes[voterAddr].interactions) ctx.nodes[voterAddr].interactions = [];
                ctx.nodes[voterAddr].interactions.push({
                    type: "VoteForArbitrator", disputeId: disputeId, votedFor: voteData.voteTarget.toLowerCase(),
                    timestamp: voteData.block.timestamp, txHash: voteData.receipt.transactionHash
                });
            }
        }
        dispute.status = "VotingCompleted";
        dispute.lastUpdateTimestamp = maxTimestamp;
        return ctx;
    });

    // 4. Select Arbitrator
    console.log(`\n4. Selecting arbitrator for Dispute ID ${disputeId}...`);
    let selectTx, selectReceipt, selectBlock, arbitratorSelectedEvent, selectedArbitratorAddress;
    let selectedArbitratorSigner = null;
    try {
        selectTx = await supplyChainNFT.connect(deployer).selectArbitrator(disputeId, gasOptions);
        selectReceipt = await selectTx.wait(1);
        selectBlock = await ethers.provider.getBlock(selectReceipt.blockNumber);
        console.log(`    Arbitrator selection process triggered. Gas Used: ${selectReceipt.gasUsed.toString()}`);

        arbitratorSelectedEvent = selectReceipt.logs?.map(log => {
            try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
        }).find(event => event?.name === "ArbitratorSelected");

        if (arbitratorSelectedEvent) {
            selectedArbitratorAddress = arbitratorSelectedEvent.args.selectedArbitrator.toLowerCase();
            console.log(`    ArbitratorSelected event processed. Selected: ${selectedArbitratorAddress}`);
        } else {
            console.warn("    WARN: ArbitratorSelected event not found. Fetching from contract state...");
            const disputeData = await supplyChainNFT.disputesData(disputeId);
            if (disputeData.selectedArbitrator !== ethers.ZeroAddress) {
                selectedArbitratorAddress = disputeData.selectedArbitrator.toLowerCase();
                console.log(`    Fallback: Directly fetched selected arbitrator: ${selectedArbitratorAddress}`);
            } else {
                throw new Error("Could not determine selected arbitrator from event or state.");
            }
        }

        if (selectedArbitratorAddress !== designatedArbitrator.address.toLowerCase()) {
             console.warn(`    WARN: Designated arbitrator ${designatedArbitrator.address} was NOT selected. Actual: ${selectedArbitratorAddress}.`);
        }
        selectedArbitratorSigner = signers.find(s => s.address.toLowerCase() === selectedArbitratorAddress);
        if (!selectedArbitratorSigner) {
            throw new Error(`Could not find signer for selected arbitrator ${selectedArbitratorAddress}`);
        }

    } catch (error) {
        console.error("  ERROR selecting arbitrator:", error);
        process.exit(1); // Exit if arbitrator cannot be selected
    }

    // Update context with selected arbitrator
    currentContext = readAndUpdateContext(ctx => {
        const dispute = ctx.disputes[disputeId];
        if (!dispute) return ctx;
        dispute.selectedArbitrator = selectedArbitratorAddress;
        dispute.status = "ArbitratorSelected";
        dispute.lastUpdateTimestamp = selectBlock.timestamp;
        if (ctx.nodes && ctx.nodes[deployer.address.toLowerCase()]) {
            if (!ctx.nodes[deployer.address.toLowerCase()].interactions) ctx.nodes[deployer.address.toLowerCase()].interactions = [];
            ctx.nodes[deployer.address.toLowerCase()].interactions.push({
                type: "SelectArbitrator", disputeId: disputeId, selected: selectedArbitratorAddress,
                timestamp: selectBlock.timestamp, txHash: selectReceipt.transactionHash
            });
        }
        return ctx;
    });

    // 5. Arbitrator Makes Decision
    console.log(`\n5. Selected Arbitrator (${selectedArbitratorAddress}) making decision for Dispute ID ${disputeId}...`);
    const resolutionOutcome = 1; // 1: Favor Disputer (Retailer), meaning full refund and product return
    const resolutionDetails = "Arbitrator decision: Full refund to disputer, product to be returned to Manufacturer.";
    const resolutionData = {
        timestamp: new Date().toISOString(),
        disputeId: disputeId,
        arbitrator: selectedArbitratorAddress,
        decision: resolutionDetails,
        outcome: resolutionOutcome,
        actionsRequired: [
            `Disputing Party (${disputingPartySigner.address}) to have product (Token ID: ${tokenIdToDispute}) returned to Manufacturer (${manufacturer.address})`,
            `Manufacturer (${manufacturer.address}) to fund, and contract to issue, full refund to Disputing Party (${disputingPartySigner.address}).`
        ],
        settlementTerms: "NFT return and full refund to be enforced on-chain."
    };
    const resolutionDataString = JSON.stringify(resolutionData);
    console.log(`    Decision: ${resolutionDetails}`);

    let decideTx, decideReceipt, decideBlock;
    try {
        decideTx = await supplyChainNFT.connect(selectedArbitratorSigner).recordDecision(disputeId, resolutionDetails, resolutionDataString, resolutionOutcome, gasOptions);
        decideReceipt = await decideTx.wait(1);
        decideBlock = await ethers.provider.getBlock(decideReceipt.blockNumber);
        console.log(`    Dispute decision made. Gas Used: ${decideReceipt.gasUsed.toString()}`);
    } catch (error) {
        console.error("  ERROR making dispute decision:", error);
        process.exit(1);
    }

    // Update context with decision
    currentContext = readAndUpdateContext(ctx => {
        const dispute = ctx.disputes[disputeId];
        if (!dispute) return ctx;
        dispute.status = "DecisionMade";
        dispute.resolutionOutcome = resolutionOutcome;
        dispute.resolutionDetails = resolutionDetails;
        dispute.resolutionDataString = resolutionDataString;
        dispute.decisionTimestamp = decideBlock.timestamp;
        dispute.lastUpdateTimestamp = decideBlock.timestamp;
        if (ctx.products && ctx.products[tokenIdToDispute]) {
            ctx.products[tokenIdToDispute].disputeInfo.status = "DecisionMade";
            ctx.products[tokenIdToDispute].lastUpdateTimestamp = decideBlock.timestamp;
        }
        if (ctx.nodes && ctx.nodes[selectedArbitratorAddress]) {
            if (!ctx.nodes[selectedArbitratorAddress].interactions) ctx.nodes[selectedArbitratorAddress].interactions = [];
            ctx.nodes[selectedArbitratorAddress].interactions.push({
                type: "MakeDisputeDecision", disputeId: disputeId, outcome: resolutionOutcome,
                timestamp: decideBlock.timestamp, txHash: decideReceipt.transactionHash
            });
        }
        return ctx;
    });

    // 6. Enforce Decision (NFT Return and Refund)
    console.log(`\n6. Enforcing decision for Dispute ID ${disputeId}...`);
    
    if (resolutionOutcome === 1) { // Favor Disputer - requires NFT return to Manufacturer and refund to Disputer
        console.log(`    Outcome favors Disputer (Retailer). Processing NFT return to Manufacturer and refund enforcement...`);
        
        // 6.1 Enforce NFT Return to Manufacturer
        console.log(`\n  6.1. Enforcing NFT return to Manufacturer for Dispute ID ${disputeId}...`);
        const returnToAddress = manufacturer.address;
        let nftReturnEnforced = false;
        
        try {
            const returnTx = await supplyChainNFT.connect(selectedArbitratorSigner).enforceNFTReturn(disputeId, returnToAddress, gasOptions);
            const returnReceipt = await returnTx.wait(1);
            console.log(`    Transaction to enforce NFT return sent. Gas Used: ${returnReceipt.gasUsed.toString()}`);
            
            // Check for NFTReturnEnforced event
            let nftReturnEnforcedEvent = returnReceipt.logs?.map(log => {
                try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
            }).find(event => event?.name === "NFTReturnEnforced");

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
                nftReturnEnforced = true;
            }
        } catch (error) {
            console.error("  ERROR enforcing NFT return:", error);
        }

        // 6.2 Enforce Refund
        console.log(`\n  6.2. Enforcing refund for Dispute ID ${disputeId}...`);
        let refundEnforced = false;
        
        // Calculate refund amount - using fixed price for Product01 similar to old logic
        const originalPriceInWei = ethers.parseUnits("0.1", "ether"); // 0.1 POL for Product01
        const compensationFeePOL = ethers.parseUnits("0.01", "ether"); // 0.01 POL compensation
        const refundAmount = originalPriceInWei + compensationFeePOL;
        const refundTo = disputingPartySigner.address; // Retailer gets refund
        const refundFrom = manufacturer.address; // Manufacturer funds the refund
        
        console.log(`    Calculated refund amount: ${ethers.formatEther(refundAmount)} POL (${ethers.formatEther(originalPriceInWei)} original + ${ethers.formatEther(compensationFeePOL)} compensation)`);
        
        // Manufacturer deposits funds for refund
        try {
            console.log(`    Manufacturer (${manufacturer.address}) depositing ${ethers.formatEther(refundAmount)} POL for refund...`);
            const depositTx = await supplyChainNFT.connect(manufacturer).depositDisputeFunds({ ...gasOptions, value: refundAmount });
            const depositReceipt = await depositTx.wait(1);
            console.log(`    Manufacturer deposited funds successfully. Gas Used: ${depositReceipt.gasUsed.toString()}`);

            let fundsDepositedEvent = depositReceipt.logs?.map(log => {
                try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
            }).find(event => event?.name === "FundsDeposited");
            
            if(fundsDepositedEvent) {
                console.log(`      FundsDeposited: Depositor=${fundsDepositedEvent.args.depositor}, Amount=${ethers.formatEther(fundsDepositedEvent.args.amount)} POL`);
            }
        } catch (e) {
            console.error(`    ERROR: Manufacturer failed to deposit funds: ${e.message}`);
            console.log("    Skipping refund enforcement due to deposit failure.");
        }

        // Check contract balance and enforce refund
        const contractBalanceBeforeRefund = await ethers.provider.getBalance(contractAddress);
        console.log(`    Contract balance before attempting refund: ${ethers.formatEther(contractBalanceBeforeRefund)} POL`);

        if (contractBalanceBeforeRefund >= refundAmount) {
            try {
                console.log(`    Arbitrator enforcing Refund of ${ethers.formatEther(refundAmount)} POL to Disputing Party (${refundTo})...`);
                console.log(`     Refund will be sourced from contract balance (notionally from Manufacturer: ${refundFrom}).`);
                
                const buyerBalanceBeforeRefund = await ethers.provider.getBalance(refundTo);

                const refundTx = await supplyChainNFT.connect(selectedArbitratorSigner).enforceRefund(disputeId, refundTo, refundFrom, refundAmount, gasOptions);
                const refundReceipt = await refundTx.wait(1);
                console.log(`    Transaction to enforce refund sent. Gas Used: ${refundReceipt.gasUsed.toString()}`);

                let refundEnforcedEvent = refundReceipt.logs?.map(log => {
                    try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
                }).find(event => event?.name === "RefundEnforced");

                if (refundEnforcedEvent) {
                    console.log(`      RefundEnforced: DisputeID=${refundEnforcedEvent.args.disputeId}, To=${refundEnforcedEvent.args.to}, From=${refundEnforcedEvent.args.from}, Amount=${ethers.formatEther(refundEnforcedEvent.args.amount)} POL`);
                } else {
                    console.warn("      WARN: Could not find RefundEnforced event in receipt.");
                }

                const buyerBalanceAfterRefund = await ethers.provider.getBalance(refundTo);
                const balanceDifference = buyerBalanceAfterRefund - buyerBalanceBeforeRefund;
                console.log(`      Balance difference for disputer: ${ethers.formatEther(balanceDifference)} POL (Expected: ~${ethers.formatEther(refundAmount)} POL)`);
                refundEnforced = true;
            } catch (error) {
                console.error("  ERROR enforcing refund:", error);
            }
        } else {
            console.error(`    ERROR: Insufficient contract balance ${ethers.formatEther(contractBalanceBeforeRefund)} POL for refund ${ethers.formatEther(refundAmount)} POL.`);
        }

        // 6.3 Conclude Dispute
        console.log(`\n  6.3. Concluding Dispute ID ${disputeId}...`);
        try {
            const concludeTx = await supplyChainNFT.connect(selectedArbitratorSigner).concludeDispute(disputeId, gasOptions);
            const concludeReceipt = await concludeTx.wait(1);
            console.log(`    Transaction to conclude dispute sent. Gas Used: ${concludeReceipt.gasUsed.toString()}`);
            
            let disputeConcludedEvent = concludeReceipt.logs?.map(log => {
                try { return supplyChainNFT.interface.parseLog(log); } catch { return null; }
            }).find(event => event?.name === "DisputeConcluded");

            if (disputeConcludedEvent) {
                console.log(`      DisputeConcluded: DisputeID=${disputeConcludedEvent.args.disputeId}, WasEnforced=${disputeConcludedEvent.args.wasEnforced}, ConcludedBy=${disputeConcludedEvent.args.concludedBy}, Timestamp=${new Date(Number(disputeConcludedEvent.args.timestamp) * 1000).toISOString()}`);
            } else {
                console.warn("      WARN: Could not find DisputeConcluded event in receipt.");
            }
        } catch (error) {
            console.error("  ERROR concluding dispute:", error);
        }

        // Update enforcement status
        const enforced = nftReturnEnforced && refundEnforced;
        const enforceBlock = await ethers.provider.getBlock('latest');
        
        // Update context with enforcement status
        currentContext = readAndUpdateContext(ctx => {
            const dispute = ctx.disputes[disputeId];
            if (!dispute) return ctx;
            dispute.status = enforced ? "Enforced" : "PartiallyEnforced";
            dispute.nftReturnEnforced = nftReturnEnforced;
            dispute.refundEnforced = refundEnforced;
            dispute.enforced = enforced;
            dispute.enforcedTimestamp = enforceBlock.timestamp;
            dispute.lastUpdateTimestamp = enforceBlock.timestamp;
            if (ctx.products && ctx.products[tokenIdToDispute]) {
                ctx.products[tokenIdToDispute].disputeInfo.status = dispute.status;
                if (nftReturnEnforced) {
                    ctx.products[tokenIdToDispute].currentOwnerAddress = manufacturer.address.toLowerCase();
                    ctx.products[tokenIdToDispute].status = "ReturnedToManufacturer_Dispute";
                    // Add history entry for return
                    if (!ctx.products[tokenIdToDispute].history) ctx.products[tokenIdToDispute].history = [];
                    ctx.products[tokenIdToDispute].history.push({
                        event: "NFTReturned_Dispute", actor: selectedArbitratorSigner.address.toLowerCase(),
                        from: retailer.address.toLowerCase(), 
                        to: manufacturer.address.toLowerCase(),
                        disputeId: disputeId,
                        timestamp: enforceBlock.timestamp,
                        details: `NFT returned to Manufacturer due to dispute ${disputeId} resolution.`,
                        txHash: "enforcement_tx"
                    });
                }
                ctx.products[tokenIdToDispute].lastUpdateTimestamp = enforceBlock.timestamp;
            }
            return ctx;
        });

        console.log(`\n--- Dispute ID ${disputeId} fully processed and concluded. ---`);
    } else {
        console.log("    Outcome does not require automated enforcement via this script.");
    }

    console.log("\n--- 06: Dispute Resolution Scenario for Product01 Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });

