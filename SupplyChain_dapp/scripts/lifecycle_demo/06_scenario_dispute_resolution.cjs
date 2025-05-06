const { ethers } = require("hardhat");
const fs = require("fs");
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

async function main() {
    console.log("--- Starting 06: Dispute Resolution Scenario ---");

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
    // transporter1Acc = signers[2]; transporter2Acc = signers[3]; batchProposerAcc = signers[4];
    // retailerAcc = signers[5]; buyer1Acc = signers[6]; arbitratorAcc = signers[7];

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }

    const deployer = signers[0];
    const manufacturer = signers[1];
    const buyer1 = signers[6]; // This buyer purchased Product 1
    const designatedArbitrator = signers[7]; // Configured as Arbitrator in script 01
    // For candidate voting, we can use other configured accounts like transporters or retailer
    const arbitratorCandidate1 = signers[2]; // transporter1Acc
    const arbitratorCandidate2 = signers[3]; // transporter2Acc

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // --- Scenario: Dispute for Product 1 (purchased by buyer1) ---
    // Let's assume Product 1 (tokenId1) has an issue after purchase and transport.
    const product1Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_001");
    if (!product1Info || !product1Info.tokenId || !product1Info.currentOwnerAddress) {
        console.error("Product 1 details or owner not found in context. Run previous scripts.");
        process.exit(1);
    }
    const tokenIdToDispute = product1Info.tokenId;
    const disputingPartySigner = signers.find(s => s.address === product1Info.currentOwnerAddress); // Buyer1

    if (!disputingPartySigner) {
        console.error(`Could not find signer for disputing party ${product1Info.currentOwnerAddress}`);
        process.exit(1);
    }

    console.log(`\n--- Disputing Product ${product1Info.uniqueProductID} (Token ID: ${tokenIdToDispute}) ---`);
    console.log(`Disputing Party (Current Owner): ${disputingPartySigner.address}`);
    const disputeReason = "Product received damaged, quality not as expected.";
    const evidenceCID = "ipfs://QmEvidenceForDamagedProduct"; // Simulated CID

    console.log(`  Opening dispute for Token ID ${tokenIdToDispute} by ${disputingPartySigner.address}...`);
    let tx = await supplyChainNFT.connect(disputingPartySigner).openDispute(tokenIdToDispute, disputeReason, evidenceCID);
    let receipt = await tx.wait(1);
    console.log(`    DisputeOpened event emitted. Gas Used: ${receipt.gasUsed.toString()}`);

    let openDisputeEvent;
    for (const log of receipt.logs) {
        try {
            const parsedLog = supplyChainNFT.interface.parseLog(log);
            if (parsedLog && parsedLog.name === "DisputeOpened") {
                openDisputeEvent = parsedLog;
                break;
            }
        } catch (e) { /* Ignore */ }
    }
    if (!openDisputeEvent) throw new Error("DisputeOpened event not found.");
    const disputeId = openDisputeEvent.args.disputeId;
    console.log(`    Dispute ID: ${disputeId.toString()}`);

    // Arbitrator Candidate Proposal and Voting (Simplified)
    // In a real scenario, candidates might be proposed by different parties or self-nominate.
    // For demo, let deployer propose candidates.
    console.log(`  Proposing arbitrator candidates for Dispute ID ${disputeId.toString()}...`);
    await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate1.address)).wait();
    console.log(`    Proposed Candidate 1: ${arbitratorCandidate1.address}`);
    await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, arbitratorCandidate2.address)).wait();
    console.log(`    Proposed Candidate 2: ${arbitratorCandidate2.address}`);
    await (await supplyChainNFT.connect(deployer).proposeArbitratorCandidate(disputeId, designatedArbitrator.address)).wait(); // The one we want to win
    console.log(`    Proposed Candidate 3 (Designated): ${designatedArbitrator.address}`);

    console.log("  Voting for arbitrator candidates...");
    // Let's have a few parties vote for the designatedArbitrator to ensure selection
    // Parties involved in the dispute (buyer, manufacturer) and other stakeholders can vote.
    const voters = [buyer1, manufacturer, signers[2], signers[3]]; // Buyer, Manufacturer, T1, T2
    for (const voter of voters) {
        // Simple voting: everyone votes for the designatedArbitrator for demo simplicity
        console.log(`    Voter ${voter.address} voting for ${designatedArbitrator.address}...`);
        try {
            tx = await supplyChainNFT.connect(voter).voteForArbitrator(disputeId, designatedArbitrator.address);
            receipt = await tx.wait(1);
            console.log(`      Vote cast. Gas Used: ${receipt.gasUsed.toString()}`);
        } catch (e) {
            console.warn(`      WARN: Voter ${voter.address} could not vote (perhaps already voted or not eligible): ${e.message}`);
        }
    }

    console.log(`  Selecting arbitrator for Dispute ID ${disputeId.toString()}...`);
    tx = await supplyChainNFT.connect(deployer).selectArbitrator(disputeId);
    receipt = await tx.wait(1);
    console.log(`    ArbitratorSelectionAttempted event emitted. Gas Used: ${receipt.gasUsed.toString()}`);

    let arbitratorSelectedEvent;
    for (const log of receipt.logs) {
        try {
            const parsedLog = supplyChainNFT.interface.parseLog(log);
            if (parsedLog && parsedLog.name === "ArbitratorSelected") {
                arbitratorSelectedEvent = parsedLog;
                break;
            }
        } catch (e) { /* Ignore */ }
    }
    if (!arbitratorSelectedEvent) {
        console.error("    ERROR: ArbitratorSelected event not found. Check voting or selection logic.");
        // Potentially, the designated arbitrator didn't win, or selection failed.
        // For the demo, we assume it works with the designated one.
    } else {
        const selectedArbitrator = arbitratorSelectedEvent.args.selectedArbitrator;
        console.log(`    Arbitrator Selected: ${selectedArbitrator} (Expected: ${designatedArbitrator.address})`);
        if (selectedArbitrator !== designatedArbitrator.address) {
            console.warn("    WARN: Designated arbitrator was not selected. The demo might not proceed as planned with this arbitrator.");
        }

        // Resolve Dispute (by the selected arbitrator)
        if (selectedArbitrator === designatedArbitrator.address) {
            console.log(`  Resolving Dispute ID ${disputeId.toString()} by selected arbitrator ${designatedArbitrator.address}...`);
            const resolutionDetails = "Arbitrator decision: Partial refund issued, product to be returned.";
            const resolutionCID = "ipfs://QmResolutionDetailsForDispute"; // Simulated CID
            const resolutionOutcome = 1; // Example: 0=Dismissed, 1=FavorPlaintiff, 2=FavorDefendant, 3=Partial

            tx = await supplyChainNFT.connect(designatedArbitrator).resolveDispute(disputeId, resolutionDetails, resolutionCID, resolutionOutcome);
            receipt = await tx.wait(1);
            console.log(`    DisputeResolved event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
            console.log(`      Resolution: ${resolutionDetails}, Outcome: ${resolutionOutcome}`);
        } else {
            console.log(`  Skipping dispute resolution as the expected arbitrator (${designatedArbitrator.address}) was not selected. Selected: ${selectedArbitrator}`);
        }
    }

    console.log("--- 06: Dispute Resolution Scenario Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

