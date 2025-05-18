const { ethers } = require("hardhat");
const fs =require("fs");
const path = require("path");
const dotenv = require('dotenv');

// Load environment variables from w3storage-upload-script directory
dotenv.config({ path: path.resolve(__dirname, '../../w3storage-upload-script/ifps_qr.env') });

// Imports for new w3up-client
const Client = require("@web3-storage/w3up-client");
const { StoreMemory } = require("@web3-storage/w3up-client/stores/memory");
const Proof = require("@web3-storage/w3up-client/proof");
const { Signer } = require("@web3-storage/w3up-client/principal/ed25519");
const { Blob, File } = require("@web-std/file");

let ipfsClientInstance; // To store the initialized client

// Helper function to initialize Web3.Storage client (w3up-client)
async function initWeb3StorageClient() {
    if (ipfsClientInstance) return ipfsClientInstance;

    const W3UP_KEY = process.env.W3UP_KEY || process.env.KEY; // KEY is used in backendListener
    const W3UP_PROOF = process.env.W3UP_PROOF || process.env.PROOF; // PROOF is used in backendListener

    if (!W3UP_KEY || !W3UP_PROOF) {
        console.warn("  WARN: W3UP_KEY or W3UP_PROOF not configured in environment variables. IPFS uploads will use placeholders.");
        return null;
    }

    try {
        const principal = Signer.parse(W3UP_KEY);
        const store = new StoreMemory();
        const client = await Client.create({ principal, store });
        const proof = await Proof.parse(W3UP_PROOF);

        const space = await client.addSpace(proof);
        await client.setCurrentSpace(space.did());
        console.log("  Web3.Storage client (w3up) initialized and space set.");
        ipfsClientInstance = client;
        return client;
    } catch (err) {
        if (err.message.includes("space already registered")) {
            console.warn("  WARN: Web3.Storage space already registered. Attempting to set current space.");
            // Re-create client to query spaces, as the first client might be in a bad state after addSpace error
            const principal = Signer.parse(W3UP_KEY);
            const store = new StoreMemory();
            const client = await Client.create({ principal, store });
            
            const spaces = await client.spaces();
            if (spaces.length > 0) {
                await client.setCurrentSpace(spaces[0].did());
                console.log(`  Web3.Storage current space set to: ${spaces[0].did()}`);
                ipfsClientInstance = client;
                return client;
            } else {
                console.error("  ERROR: Web3.Storage space already registered, but no spaces found to set as current.");
                return null;
            }
        } else {
            console.error("  ERROR initializing Web3.Storage client (w3up):", err);
            return null;
        }
    }
}

// Updated helper function to upload data to IPFS via w3up-client
async function uploadToIPFS(data, fileName) {
    const client = await initWeb3StorageClient();
    if (!client) {
        console.warn(`  WARN: IPFS client (w3up) not available. Using placeholder CID for ${fileName}.`);
        return `ipfs://placeholder_for_${fileName}_due_to_client_init_failure`;
    }

    console.log(`  Uploading ${fileName} to IPFS via w3up-client...`);
    try {
        const dataBlob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const file = new File([dataBlob], fileName);
        const cid = await client.uploadFile(file);
        console.log(`    Successfully uploaded ${fileName}, CID: ${cid.toString()}`);
        return `ipfs://${cid.toString()}`;
    } catch (error) {
        console.error(`    Error uploading ${fileName} to IPFS (w3up):`, error);
        return `ipfs://placeholder_for_${fileName}_due_to_upload_error`;
    }
}

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

async function main() {
    console.log("--- Starting 06: Dispute Resolution Scenario ---");

    // Initialize client once at the start if preferred, or let uploadToIPFS handle it.
    // await initWeb3StorageClient(); // Optional: pre-initialize

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
    
    // Simulate creating evidence data
    const evidenceData = {
        timestamp: new Date().toISOString(),
        disputeReason: disputeReason,
        productID: product1Info.uniqueProductID,
        tokenId: tokenIdToDispute.toString(),
        images: ["image_proof1.jpg", "video_proof.mp4"], // Example file names
        description: "Detailed description of the damage and discrepancy."
    };
    // Upload evidence to IPFS
    const evidenceCID = await uploadToIPFS(evidenceData, `evidence_dispute_${disputeIdCounter}_${tokenIdToDispute}.json`);
    console.log(`  Generated Evidence CID: ${evidenceCID}`);
    disputeIdCounter++; // Increment for unique file names if multiple disputes run in one script exec

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
            const resolutionDetailsText = "Arbitrator decision: Partial refund issued, product to be returned.";
            const resolutionOutcome = 1; // Example: 0=Dismissed, 1=FavorPlaintiff, 2=FavorDefendant, 3=Partial

            // Simulate creating resolution data
            const resolutionData = {
                timestamp: new Date().toISOString(),
                disputeId: disputeId.toString(),
                arbitrator: designatedArbitrator.address,
                decision: resolutionDetailsText,
                outcome: resolutionOutcome,
                actionsRequired: ["Buyer to return product to Manufacturer", "Manufacturer to issue 50% refund upon receipt"],
                settlementTerms: "As per arbitrator ruling."
            };
            // Upload resolution data to IPFS
            const resolutionCID = await uploadToIPFS(resolutionData, `resolution_dispute_${disputeId.toString()}.json`);
            console.log(`  Generated Resolution CID: ${resolutionCID}`);

            tx = await supplyChainNFT.connect(designatedArbitrator).resolveDispute(disputeId, resolutionDetailsText, resolutionCID, resolutionOutcome);
            receipt = await tx.wait(1);
            console.log(`    DisputeResolved event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
            console.log(`      Resolution: ${resolutionDetailsText}, Outcome: ${resolutionOutcome}`);
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

