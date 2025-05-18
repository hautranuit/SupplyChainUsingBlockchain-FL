const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run previous lifecycle scripts first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

// Helper function to simulate a delay
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    console.log("--- Starting 04: Transport, Batch Processing, and Finalization Scenario ---");

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    let productDetails = context.productDetails; 

    if (!contractAddress || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and productDetails are present.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    // Based on 01_deploy_and_configure.cjs:
    // signers[0] = deployer 
    // signers[1] = manufacturerAcc 
    // signers[2] = transporter1Acc 
    // signers[3] = transporter2Acc 
    // signers[4] = transporter3Acc 
    // signers[5] = retailerAcc 
    // signers[6] = buyer1Acc 
    // signers[7] = arbitratorAcc

    if (signers.length < 8) { // Adjusted from 10 to 8
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.cjs for all roles.");
        process.exit(1);
    }

    const deployer = signers[0];
    const manufacturer = signers[1]; // Is a Primary Node
    const transporter1 = signers[2]; // Is a Secondary Node
    const transporter2 = signers[3]; // Is a Secondary Node
    const transporter3 = signers[4]; // Is a Secondary Node
    const retailer = signers[5];     // Is a Primary Node
    const buyer1 = signers[6];       // Is a Secondary Node
    // const arbitrator = signers[7]; // Is a Primary Node, not directly used for batch validation in this script flow but available

    // Validators for batch processing will be manufacturer and retailer (as examples of PNs)
    const validator1 = manufacturer; 
    const validator2 = retailer;

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract SupplyChainNFT.");

    // Define gas options to be used for transactions
    const gasOptions = { 
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'), // Adjusted based on Amoy
        maxFeePerGas: ethers.parseUnits('40', 'gwei')         // Adjusted based on Amoy
    };
    console.log("Using gas options:", gasOptions);

    // --- Scenario 1: Product 1 (DEMO_PROD_001) - Sequential Transport & Finalization ---
    console.log("\n--- Scenario 1: Product 1 (DEMO_PROD_001) ---");
    const product1Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_001");
    if (!product1Info || !product1Info.tokenId) {
        console.error("Product 1 (DEMO_PROD_001) details not found. Run previous scripts.");
        process.exit(1);
    }
    const tokenId1 = product1Info.tokenId;
    console.log(`Processing Product 1 (Token ID: ${tokenId1})`);

    if (product1Info.productStatus !== "CollateralDeposited") {
        console.error(`Product 1 is not in 'CollateralDeposited' state. Current state: ${product1Info.productStatus}. Aborting transport.`);
        process.exit(1);
    }
    
    // 1. PRODUCT 1: Sequential Transport (Manufacturer -> T1 -> T2 -> T3 -> Buyer1)
    // ###########################################################################
    console.log("\\n--- Product 1: Sequential Transport and Finalization ---");
    let currentOwnerP1Signer = manufacturer; // Manufacturer is the initial owner and transporter

    // 1.1 Manufacturer starts transport for Product 1, detailing the full journey
    console.log(`  1.1 Manufacturer (${manufacturer.address}) starting transport for Token ID ${tokenId1}. Full Journey: M -> T1 -> T2 -> T3 -> Buyer1's Location.`);
    const p1_transportLegs = [transporter1.address, transporter2.address, transporter3.address]; // All transporter legs
    const p1_startLocation = "Manufacturer Site";
    const p1_endLocation = "Buyer 1's Location (via T1, T2, T3)"; // Final conceptual destination
    const p1_distance = 450; // Estimated total distance for all legs (e.g., 150 + 150 + 150)
    let tx = await supplyChainNFT.connect(manufacturer).startTransport(tokenId1, p1_transportLegs, p1_startLocation, p1_endLocation, p1_distance, gasOptions);
    let receipt = await tx.wait(1);
    console.log(`      TransportStarted event emitted. Gas: ${receipt.gasUsed.toString()}. (Backend listener should process)`);
    
    // Manufacturer transfers NFT to Transporter 1
    console.log(`      Manufacturer (${manufacturer.address}) transferring Token ID ${tokenId1} to Transporter 1 (${transporter1.address}).`);
    tx = await supplyChainNFT.connect(manufacturer).transferFrom(manufacturer.address, transporter1.address, tokenId1, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      NFT ${tokenId1} transferred from Manufacturer to Transporter 1. Gas: ${receipt.gasUsed.toString()}. Event: Transfer.`);
    currentOwnerP1Signer = transporter1;
    console.log(`      Product 1 (Token ID: ${tokenId1}) now owned by Transporter 1 (${transporter1.address}).`);
    await delay(2000);

    // 1.2 Transporter 1 completes their leg, transfers to Transporter 2
    console.log(`  1.2 Transporter 1 (${transporter1.address}) is now in possession of Token ID ${tokenId1}.`);
    // console.log(`  1.2 Transporter 1 (${transporter1.address}) completing transport for Token ID ${tokenId1}.`);
    // tx = await supplyChainNFT.connect(currentOwnerP1Signer).completeTransport(tokenId1, gasOptions);
    // receipt = await tx.wait(1);
    // console.log(`      TransportCompleted (leg 1) event emitted. Gas: ${receipt.gasUsed.toString()}. (Backend listener should process)`);
    
    console.log(`      Transporter 1 (${transporter1.address}) transferring Token ID ${tokenId1} to Transporter 2 (${transporter2.address}).`);
    tx = await supplyChainNFT.connect(currentOwnerP1Signer).transferFrom(transporter1.address, transporter2.address, tokenId1, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      NFT ${tokenId1} transferred from Transporter 1 to Transporter 2. Gas: ${receipt.gasUsed.toString()}. Event: Transfer.`);
    currentOwnerP1Signer = transporter2;
    console.log(`      Product 1 (Token ID: ${tokenId1}) now owned by Transporter 2 (${transporter2.address}).`);
    await delay(2000);

    // 1.3 Transporter 2 completes their leg, transfers to Transporter 3
    console.log(`  1.3 Transporter 2 (${transporter2.address}) is now in possession of Token ID ${tokenId1}.`);
    // console.log(`  1.3 Transporter 2 (${transporter2.address}) completing transport for Token ID ${tokenId1}.`);
    // tx = await supplyChainNFT.connect(currentOwnerP1Signer).completeTransport(tokenId1, gasOptions);
    // receipt = await tx.wait(1);
    // console.log(`      TransportCompleted (leg 2) event emitted. Gas: ${receipt.gasUsed.toString()}. (Backend listener should process)`);

    console.log(`      Transporter 2 (${transporter2.address}) transferring Token ID ${tokenId1} to Transporter 3 (${transporter3.address}).`);
    tx = await supplyChainNFT.connect(currentOwnerP1Signer).transferFrom(transporter2.address, transporter3.address, tokenId1, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      NFT ${tokenId1} transferred from Transporter 2 to Transporter 3. Gas: ${receipt.gasUsed.toString()}. Event: Transfer.`);
    currentOwnerP1Signer = transporter3;
    console.log(`      Product 1 (Token ID: ${tokenId1}) now owned by Transporter 3 (${transporter3.address}).`);
    await delay(2000);

    // 1.4 Transporter 3 completes their leg (final delivery to buyer's proximity)
    console.log(`  1.4 Transporter 3 (${transporter3.address}) completing final transport leg for Token ID ${tokenId1}.`);
    tx = await supplyChainNFT.connect(currentOwnerP1Signer).completeTransport(tokenId1, gasOptions); // currentOwnerP1Signer is transporter3
    receipt = await tx.wait(1);
    console.log(`      TransportCompleted (leg 3 - final) event emitted. Gas: ${receipt.gasUsed.toString()}. (Backend listener should process)`);
    // Ownership is with Transporter 3. The actual NFT transfer to buyer happens in finalizePurchaseAndTransfer.
    console.log(`      (Simulation) Product 1 delivered to buyer's (${buyer1.address}) proximity. NFT still owned by ${transporter3.address}.`);
    // Update product status in demo_context.json (optional, for script state tracking)
    product1Info.productStatus = "DeliveredToBuyerProximity"; // Or a more suitable status from your contract
    await delay(2000);

    // 1.5 Buyer (or Deployer on behalf of Buyer) confirms delivery and finalizes purchase for Product 1
    console.log(`  1.5 Buyer (${buyer1.address}) or Deployer confirming delivery and finalizing purchase for Product 1 (Token ID: ${tokenId1}).`);
    
    // Pre-finalization state logging
    console.log("    --- Pre-Finalization State ---");
    console.log(`    Token ID: ${tokenId1}`);
    console.log(`    Product Seller: ${product1Info.sellerAddress}`);
    console.log(`    Buyer: ${buyer1.address}`);
    const currentOwnerOfToken1 = await supplyChainNFT.ownerOf(tokenId1);
    console.log(`    Current Owner of Token ${tokenId1}: ${currentOwnerOfToken1}`);
    const purchaseInfoP1 = await supplyChainNFT.getPurchaseInfo(tokenId1); // Assuming getPurchaseInfo is on supplyChainNFT (NFTCore)
    console.log(`    Purchase Info for Token ${tokenId1}:`);
    console.log(`      Seller: ${purchaseInfoP1.seller}`);
    console.log(`      Buyer: ${purchaseInfoP1.buyer}`);
    console.log(`      Price: ${ethers.formatEther(purchaseInfoP1.price)} ETH`);
    console.log(`      Collateral: ${ethers.formatEther(purchaseInfoP1.collateral)} ETH`);
    console.log(`      Status (Enum): ${purchaseInfoP1.status}`); // 0: Empty, 1: Listed, 2: Initiated, 3: CollateralDeposited, 4: InTransit, 5: TransportCompleted, 6: Complete, 7: Cancelled
    const marketplaceBalance = await ethers.provider.getBalance(marketplace.target);
    console.log(`    Marketplace Contract (${marketplace.target}) Balance: ${ethers.formatEther(marketplaceBalance)} ETH`);
    const buyer1Balance = await ethers.provider.getBalance(buyer1.address);
    console.log(`    Buyer (${buyer1.address}) Balance: ${ethers.formatEther(buyer1Balance)} ETH`);
    const sellerP1Balance = await ethers.provider.getBalance(product1Info.sellerAddress);
    console.log(`    Seller (${product1Info.sellerAddress}) Balance: ${ethers.formatEther(sellerP1Balance)} ETH`);
    const meetsIncentiveCriteriaP1 = true; // Example: assume criteria met for transporter incentive
    console.log(`    Meets Incentive Criteria: ${meetsIncentiveCriteriaP1}`);
    console.log("    --- End Pre-Finalization State ---");

    // In a real scenario, the buyer would call this. For the script, deployer can do it.
    // The `meetsIncentiveCriteria` would be determined by off-chain logic or specific conditions.
    tx = await supplyChainNFT.connect(buyer1).confirmDeliveryAndFinalize(tokenId1, meetsIncentiveCriteriaP1, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      PaymentAndTransferCompleted and PurchaseStatusUpdated events expected. Gas: ${receipt.gasUsed.toString()}. (Backend listener should process)`);
    console.log(`      Product 1 (Token ID: ${tokenId1}) ownership officially transferred to buyer ${buyer1.address}. Payment released to seller ${product1Info.sellerAddress}.`);
    product1Info.productStatus = "Completed"; // Update local status
    product1Info.currentOwner = buyer1.address;
    
    console.log("--- Scenario 1: Product 1 Completed ---");
    await delay(5000); 

    // --- Scenario 2: Product 2 (DEMO_PROD_002) & Product 3 (DEMO_PROD_003) - Batched Transport & Finalization ---
    console.log("\n--- Scenario 2: Product 2 & 3 - Batched Transport to Retailer ---");
    const product2Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_002");
    const product3Info = productDetails.find(p => p.uniqueProductID === "DEMO_PROD_003");

    if (!product2Info || !product2Info.tokenId || !product3Info || !product3Info.tokenId) {
        console.error("Product 2 or 3 details not found. Run previous scripts.");
        process.exit(1);
    }
    const tokenId2 = product2Info.tokenId;
    const tokenId3 = product3Info.tokenId;
    console.log(`Processing Product 2 (Token ID: ${tokenId2}) and Product 3 (Token ID: ${tokenId3})`);

    if (product2Info.productStatus !== "CollateralDeposited" || product3Info.productStatus !== "CollateralDeposited") {
        console.error(`Product 2 or 3 not in 'CollateralDeposited' state. P2: ${product2Info.productStatus}, P3: ${product3Info.productStatus}. Aborting.`);
        process.exit(1);
    }
    if (product2Info.pendingBuyerAddress.toLowerCase() !== retailer.address.toLowerCase() || 
        product3Info.pendingBuyerAddress.toLowerCase() !== retailer.address.toLowerCase()) {
        console.error(`Error: Mismatch in pending buyer for Product 2 or 3. Expected retailer ${retailer.address}. P2 Buyer: ${product2Info.pendingBuyerAddress}, P3 Buyer: ${product3Info.pendingBuyerAddress}`);
        process.exit(1);
    }

    // 2.1 Manufacturer starts transport for Product 2 & 3 individually
    console.log(`  2.1.1 Manufacturer (${manufacturer.address}) starting transport for Product 2 (Token ID: ${tokenId2}). Legs: T1, T2, Retailer.`);
    const transportLegsP2 = [transporter1.address, transporter2.address, retailer.address]; 
    const startLocationP2 = "Manufacturer's Bulk Warehouse";
    const endLocationP2 = "Retailer Central Warehouse"; // Changed from retailer.address to a string
    const distanceP2 = 700; 
    tx = await supplyChainNFT.connect(manufacturer).startTransport(tokenId2, transportLegsP2, startLocationP2, endLocationP2, distanceP2, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P2: TransportStarted. Gas: ${receipt.gasUsed.toString()}. (Backend listener)`);
    
    console.log(`      Transferring Product 2 (Token ID: ${tokenId2}) from Manufacturer to Transporter 1 (${transporter1.address})`);
    tx = await supplyChainNFT.connect(manufacturer).transferFrom(manufacturer.address, transporter1.address, tokenId2, gasOptions);
    await tx.wait(1);
    console.log(`      Product 2 now owned by Transporter 1.`);
    await delay(1000);

    console.log(`  2.1.2 Manufacturer (${manufacturer.address}) starting transport for Product 3 (Token ID: ${tokenId3}). Legs: T1, T2, Retailer.`);
    const transportLegsP3 = [transporter1.address, transporter2.address, retailer.address];
    const startLocationP3 = "Manufacturer's Bulk Warehouse";
    const endLocationP3 = "Retailer Central Warehouse"; // Changed from retailer.address to a string
    const distanceP3 = 720; 
    tx = await supplyChainNFT.connect(manufacturer).startTransport(tokenId3, transportLegsP3, startLocationP3, endLocationP3, distanceP3, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P3: TransportStarted. Gas: ${receipt.gasUsed.toString()}. (Backend listener)`);

    console.log(`      Transferring Product 3 (Token ID: ${tokenId3}) from Manufacturer to Transporter 1 (${transporter1.address})`);
    tx = await supplyChainNFT.connect(manufacturer).transferFrom(manufacturer.address, transporter1.address, tokenId3, gasOptions);
    await tx.wait(1);
    console.log(`      Product 3 now owned by Transporter 1.`);
    await delay(2000);

    // 2.2 Batch Transfer: Transporter 1 -> Transporter 2 (for Product 2 & 3)
    console.log(`  2.2 Batch Transfer: Transporter 1 (${transporter1.address}) to Transporter 2 (${transporter2.address}) for P2 & P3.`);
    const batchTxData_T1_T2 = [
        { from: transporter1.address, to: transporter2.address, tokenId: tokenId2 },
        { from: transporter1.address, to: transporter2.address, tokenId: tokenId3 }
    ];
    
    console.log(`      Proposing batch from T1 to T2 by ${transporter1.address}...`);
    tx = await supplyChainNFT.connect(transporter1).proposeBatch(batchTxData_T1_T2, gasOptions); 
    receipt = await tx.wait(1);
    const proposeEvent_T1_T2_log = receipt.logs.find(log => log.fragment && log.fragment.name === "BatchProposed");
    if (!proposeEvent_T1_T2_log) throw new Error("BatchProposed event not found for T1->T2 batch");
    const batchId_T1_T2 = proposeEvent_T1_T2_log.args.batchId;
    console.log(`      Batch ${batchId_T1_T2} proposed. Gas: ${receipt.gasUsed.toString()}. Validators: ${proposeEvent_T1_T2_log.args.selectedValidators.join(', ')}`);
    await delay(1000);

    console.log(`      Validating Batch ${batchId_T1_T2} by Validator1 (${validator1.address}) and Validator2 (${validator2.address})...`);
    // Note: The contract internally selected validators. Here we are simulating that 
    // validator1 and validator2 (who are PNs) are among those selected and are performing the validation.
    // The script logs which validators were *actually* selected by the contract in the BatchProposed event.
    tx = await supplyChainNFT.connect(validator1).validateBatch(batchId_T1_T2, true, gasOptions); 
    await tx.wait(1);
    console.log(`        Validator1 (${validator1.address}) validated.`);
    tx = await supplyChainNFT.connect(validator2).validateBatch(batchId_T1_T2, true, gasOptions); 
    await tx.wait(1);
    console.log(`        Validator2 (${validator2.address}) validated.`);
    await delay(1000);

    console.log(`      Committing Batch ${batchId_T1_T2} by ${transporter1.address}...`);
    tx = await supplyChainNFT.connect(transporter1).commitBatch(batchId_T1_T2, gasOptions); 
    receipt = await tx.wait(1);
    console.log(`      Batch ${batchId_T1_T2} committed. Gas: ${receipt.gasUsed.toString()}. (Backend listener for individual transfers)`);
    console.log(`      Products 2 & 3 now owned by Transporter 2 (${transporter2.address})`);
    await delay(2000);

    // 2.3 Batch Transfer: Transporter 2 -> Retailer (for Product 2 & 3)
    console.log(`  2.3 Batch Transfer: Transporter 2 (${transporter2.address}) to Retailer (${retailer.address}) for P2 & P3.`);
    const batchTxData_T2_Retailer = [
        { from: transporter2.address, to: retailer.address, tokenId: tokenId2 },
        { from: transporter2.address, to: retailer.address, tokenId: tokenId3 }
    ];

    console.log(`      Proposing batch from T2 to Retailer by ${transporter2.address}...`);
    tx = await supplyChainNFT.connect(transporter2).proposeBatch(batchTxData_T2_Retailer, gasOptions);
    receipt = await tx.wait(1);
    const proposeEvent_T2_Retailer_log = receipt.logs.find(log => log.fragment && log.fragment.name === "BatchProposed");
    if (!proposeEvent_T2_Retailer_log) throw new Error("BatchProposed event not found for T2->Retailer batch");
    const batchId_T2_Retailer = proposeEvent_T2_Retailer_log.args.batchId;
    console.log(`      Batch ${batchId_T2_Retailer} proposed. Gas: ${receipt.gasUsed.toString()}. Validators: ${proposeEvent_T2_Retailer_log.args.selectedValidators.join(', ')}`);
    await delay(1000);

    console.log(`      Validating Batch ${batchId_T2_Retailer} by Validator1 (${validator1.address}) and Validator2 (${validator2.address})...`);
    tx = await supplyChainNFT.connect(validator1).validateBatch(batchId_T2_Retailer, true, gasOptions);
    await tx.wait(1);
    console.log(`        Validator1 (${validator1.address}) validated.`);
    tx = await supplyChainNFT.connect(validator2).validateBatch(batchId_T2_Retailer, true, gasOptions);
    await tx.wait(1);
    console.log(`        Validator2 (${validator2.address}) validated.`);
    await delay(1000);

    console.log(`      Committing Batch ${batchId_T2_Retailer} by ${transporter2.address}...`);
    tx = await supplyChainNFT.connect(transporter2).commitBatch(batchId_T2_Retailer, gasOptions); 
    receipt = await tx.wait(1);
    console.log(`      Batch ${batchId_T2_Retailer} committed. Gas: ${receipt.gasUsed.toString()}. (Backend listener for individual transfers)`);
    console.log(`      Products 2 & 3 now owned by Retailer (${retailer.address})`);
    await delay(2000);
    
    console.log(`  2.4 Retailer (${retailer.address}) confirming final transport completion for Product 2 (Token ID: ${tokenId2}).`);
    tx = await supplyChainNFT.connect(retailer).completeTransport(tokenId2, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P2: TransportCompleted (final by Retailer). Gas: ${receipt.gasUsed.toString()}.`);

    console.log(`  2.4 Retailer (${retailer.address}) confirming final transport completion for Product 3 (Token ID: ${tokenId3}).`);
    tx = await supplyChainNFT.connect(retailer).completeTransport(tokenId3, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P3: TransportCompleted (final by Retailer). Gas: ${receipt.gasUsed.toString()}.`);
    product3Info.productStatus = "DeliveredToBuyerProximity";
    await delay(2000);

    // 2.5 Finalize Purchases for Product 2 & 3 (Retailer is the buyer)
    console.log(`  2.5.1 Retailer (${retailer.address}) confirming delivery and finalizing purchase for Product 2 (Token ID: ${tokenId2}).`);
    const meetsIncentiveCriteriaP2 = true; // Example
    tx = await supplyChainNFT.connect(retailer).confirmDeliveryAndFinalize(tokenId2, meetsIncentiveCriteriaP2, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P2: PaymentAndTransferCompleted. Gas: ${receipt.gasUsed.toString()}. (Backend listener)`);
    console.log(`      Product 2 (Token ID: ${tokenId2}) ownership confirmed for Retailer. Payment released to seller ${product2Info.sellerAddress}.`);
    product2Info.productStatus = "Completed";
    product2Info.currentOwner = retailer.address;
    await delay(1000);

    console.log(`  2.5.2 Retailer (${retailer.address}) confirming delivery and finalizing purchase for Product 3 (Token ID: ${tokenId3}).`);
    const meetsIncentiveCriteriaP3 = true; // Example
    tx = await supplyChainNFT.connect(retailer).confirmDeliveryAndFinalize(tokenId3, meetsIncentiveCriteriaP3, gasOptions);
    receipt = await tx.wait(1);
    console.log(`      P3: PaymentAndTransferCompleted. Gas: ${receipt.gasUsed.toString()}. (Backend listener)`);
    console.log(`      Product 3 (Token ID: ${tokenId3}) ownership confirmed for Retailer. Payment released to seller ${product3Info.sellerAddress}.`);
    product3Info.productStatus = "Completed";
    product3Info.currentOwner = retailer.address;

    console.log("--- Scenario 2: Product 2 & 3 Completed ---");
    console.log("\n--- 04: Transport, Batch Processing, and Finalization Scenario Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });


