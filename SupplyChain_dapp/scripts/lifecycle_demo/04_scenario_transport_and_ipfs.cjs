const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run 03_scenario_marketplace_and_purchase.js first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

// Helper function to simulate IPFS upload and get a dummy CID
function simulateIPFSUpload(data) {
    console.log("    Simulating IPFS upload for data:", data);
    // Generate a pseudo-random CID for demo purposes
    const randomString = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    const cid = "ipfs://Qm" + randomString;
    console.log(`    Simulated IPFS CID: ${cid}`);
    return cid;
}

async function main() {
    console.log("--- Starting 04: Transport and IPFS Logging Scenarios ---");

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    let productDetails = context.productDetails; // Use let as it will be updated

    if (!contractAddress || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and productDetails are present.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    // Signers configured in script 01:
    // deployer = signers[0]; manufacturerAcc = signers[1];
    // transporter1Acc = signers[2]; transporter2Acc = signers[3]; transporter3Acc = signers[4];
    // retailerAcc = signers[5]; buyer1Acc = signers[6]; arbitratorAcc = signers[7];

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }

    const deployer = signers[0]; // Has UPDATER_ROLE for updateProductHistoryCID
    const transporter1 = signers[2];
    const transporter2 = signers[3];
    // const transporter3_secondary = signers[4]; // Not directly used as a transporter leg here, but available

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // --- Scenario 1: Transport for Product 1 (purchased by buyer1Acc in script 03) ---
    const product1Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_001");
    if (!product1Info || !product1Info.tokenId || !product1Info.currentOwnerAddress) {
        console.error("Product 1 details or owner not found in context. Run previous scripts.");
        process.exit(1);
    }
    const tokenId1 = product1Info.tokenId;
    const ownerOfToken1Signer = signers.find(s => s.address === product1Info.currentOwnerAddress);
    if (!ownerOfToken1Signer) {
        console.error(`Could not find signer for owner ${product1Info.currentOwnerAddress} of Token ID ${tokenId1}`);
        process.exit(1);
    }

    console.log(`\n--- Scenario 1: Transport for Product ${product1Info.uniqueProductID} (ID: ${tokenId1}), Owner: ${ownerOfToken1Signer.address} ---`);
    const transportLegs1 = [transporter1.address, transporter2.address];
    const startLocation1 = "Manufacturer Warehouse A";
    const endLocation1 = "Retailer Hub Z";
    const distance1 = 1200; // km

    console.log(`  Starting transport for Token ID ${tokenId1} with ${transportLegs1.length} transporters...`);
    let tx = await supplyChainNFT.connect(ownerOfToken1Signer).startTransport(tokenId1, transportLegs1, startLocation1, endLocation1, distance1);
    let receipt = await tx.wait(1);
    console.log(`    TransportStarted event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
    console.log("    ACTION: Backend listener (backendListener.js) would pick up TransportStarted event.");
    const transportLogData1 = { tokenId: tokenId1, transporters: transportLegs1, start: startLocation1, end: endLocation1, distance: distance1, timestamp: new Date().toISOString() };
    const transportLogCID1 = simulateIPFSUpload(transportLogData1);
    console.log(`    ACTION: Backend would call updateProductHistoryCID with ${transportLogCID1}`);
    tx = await supplyChainNFT.connect(deployer).updateProductHistoryCID(tokenId1, transportLogCID1);
    receipt = await tx.wait(1);
    console.log(`      Product history CID updated for transport start. Gas Used: ${receipt.gasUsed.toString()}`);
    product1Info.currentTransportCID = transportLogCID1;

    // Simulate some time passing / legs completing - contract doesn't have explicit leg completion
    console.log(`  Simulating transport progress...`);

    console.log(`  Completing transport for Token ID ${tokenId1}...`);
    tx = await supplyChainNFT.connect(ownerOfToken1Signer).completeTransport(tokenId1);
    receipt = await tx.wait(1);
    console.log(`    TransportCompleted event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
    console.log("    ACTION: Backend listener would pick up TransportCompleted event.");
    const completionLogData1 = { tokenId: tokenId1, status: "Completed", completedBy: ownerOfToken1Signer.address, timestamp: new Date().toISOString() };
    const completionLogCID1 = simulateIPFSUpload(completionLogData1);
    console.log(`    ACTION: Backend would call updateProductHistoryCID with ${completionLogCID1}`);
    tx = await supplyChainNFT.connect(deployer).updateProductHistoryCID(tokenId1, completionLogCID1);
    receipt = await tx.wait(1);
    console.log(`      Product history CID updated for transport completion. Gas Used: ${receipt.gasUsed.toString()}`);
    product1Info.finalCID = completionLogCID1;
    product1Info.transportStatus = "Completed";

    console.log(`  Conceptual QR Code for Product ${tokenId1}:`);
    console.log(`    Data to encode (e.g., latest IPFS CID): ${completionLogCID1}`);
    console.log("    ACTION: Use w3storage-upload-script/generateEncryptedQR.js to create QR.");
    console.log("    ACTION: Use w3storage-upload-script/decryptCID.js to decrypt from QR scan.");

    // --- Scenario 2: Transport for Product 2 (purchased by buyer2Acc/retailer in script 03) ---
    const product2Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_002");
    if (!product2Info || !product2Info.tokenId || !product2Info.currentOwnerAddress) {
        console.error("Product 2 details or owner not found in context. Run previous scripts.");
        process.exit(1);
    }
    const tokenId2 = product2Info.tokenId;
    const ownerOfToken2Signer = signers.find(s => s.address === product2Info.currentOwnerAddress);
     if (!ownerOfToken2Signer) {
        console.error(`Could not find signer for owner ${product2Info.currentOwnerAddress} of Token ID ${tokenId2}`);
        process.exit(1);
    }

    console.log(`\n--- Scenario 2: Transport for Product ${product2Info.uniqueProductID} (ID: ${tokenId2}), Owner: ${ownerOfToken2Signer.address} ---`);
    const transportLegs2 = [transporter1.address]; // Single transporter
    const startLocation2 = "Pharma Distribution Center";
    const endLocation2 = "City Hospital";
    const distance2 = 150; // km

    console.log(`  Starting transport for Token ID ${tokenId2} with ${transportLegs2.length} transporter...`);
    tx = await supplyChainNFT.connect(ownerOfToken2Signer).startTransport(tokenId2, transportLegs2, startLocation2, endLocation2, distance2);
    receipt = await tx.wait(1);
    console.log(`    TransportStarted event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
    const transportLogData2 = { tokenId: tokenId2, transporters: transportLegs2, start: startLocation2, end: endLocation2, distance: distance2, timestamp: new Date().toISOString() };
    const transportLogCID2 = simulateIPFSUpload(transportLogData2);
    tx = await supplyChainNFT.connect(deployer).updateProductHistoryCID(tokenId2, transportLogCID2);
    receipt = await tx.wait(1);
    console.log(`      Product history CID updated for transport start. Gas Used: ${receipt.gasUsed.toString()}`);
    product2Info.currentTransportCID = transportLogCID2;

    console.log(`  Completing transport for Token ID ${tokenId2}...`);
    tx = await supplyChainNFT.connect(ownerOfToken2Signer).completeTransport(tokenId2);
    receipt = await tx.wait(1);
    console.log(`    TransportCompleted event emitted. Gas Used: ${receipt.gasUsed.toString()}`);
    const completionLogData2 = { tokenId: tokenId2, status: "Completed", completedBy: ownerOfToken2Signer.address, timestamp: new Date().toISOString() };
    const completionLogCID2 = simulateIPFSUpload(completionLogData2);
    tx = await supplyChainNFT.connect(deployer).updateProductHistoryCID(tokenId2, completionLogCID2);
    receipt = await tx.wait(1);
    console.log(`      Product history CID updated for transport completion. Gas Used: ${receipt.gasUsed.toString()}`);
    product2Info.finalCID = completionLogCID2;
    product2Info.transportStatus = "Completed";

    // Update context
    const updatedDemoContext = {
        ...context,
        productDetails: productDetails // productDetails array was updated in place
    };
    fs.writeFileSync(path.join(__dirname, "demo_context.json"), JSON.stringify(updatedDemoContext, null, 4));
    console.log("\nDemo context updated with transport CIDs and status.");

    console.log("--- 04: Transport and IPFS Logging Scenarios Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });


