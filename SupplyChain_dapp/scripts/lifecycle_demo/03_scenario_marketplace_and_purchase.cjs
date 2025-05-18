const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read/write demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run 01_deploy_and_configure.js and 02_scenario_product_creation.cjs first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

function saveDemoContext(context) {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    fs.writeFileSync(demoContextPath, JSON.stringify(context, null, 2));
    console.log(`    demo_context.json updated successfully at ${demoContextPath}`);
}

async function main() {
    console.log("--- Starting 03: Marketplace Listing and Collateral Deposit Scenario ---");

    // Read arguments from environment variables set by the Hardhat task
    const tokenIdArg1 = process.env.SCENARIO03_TOKENID1;
    const ipfsCIDArg1 = process.env.SCENARIO03_CID1; // Renamed from initialCIDArg1
    const tokenIdArg2 = process.env.SCENARIO03_TOKENID2;
    const ipfsCIDArg2 = process.env.SCENARIO03_CID2; // Renamed from initialCIDArg2
    const tokenIdArg3 = process.env.SCENARIO03_TOKENID3;
    const ipfsCIDArg3 = process.env.SCENARIO03_CID3; // Renamed from initialCIDArg3

    if (!tokenIdArg1 || !ipfsCIDArg1 || !tokenIdArg2 || !ipfsCIDArg2 || !tokenIdArg3 || !ipfsCIDArg3) {
        console.error("Error: Missing one or more required environment variables for token IDs and IPFS CIDs.");
        console.error("Ensure SCENARIO03_TOKENID1, SCENARIO03_CID1 (IPFS CID), etc., are set by the 'run-scenario03' Hardhat task.");
        process.exit(1);
    }

    console.log("Using parameters from environment variables (via Hardhat task):");
    console.log(`  Product 1 (DEMO_PROD_001): Token ID = ${tokenIdArg1}, IPFS CID for Verification = ${ipfsCIDArg1}`);
    console.log(`  Product 2 (DEMO_PROD_002): Token ID = ${tokenIdArg2}, IPFS CID for Verification = ${ipfsCIDArg2}`);
    console.log(`  Product 3 (DEMO_PROD_003): Token ID = ${tokenIdArg3}, IPFS CID for Verification = ${ipfsCIDArg3}`);

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    let productDetails = context.productDetails;

    if (!contractAddress || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and productDetails are present from previous scripts.");
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

    const manufacturerAcc = signers[1];
    const buyer1Acc = signers[6];
    const retailerAcc = signers[5]; // Acting as Buyer 2

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress);
    console.log("Connected to contract.");

    // Define gas options to be used for transactions
    // Adjusted based on typical Amoy testnet conditions; consider making these configurable
    const gasOptions = { 
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'), // Adjusted lower, test what works
        maxFeePerGas: ethers.parseUnits('40', 'gwei') // Adjusted lower, test what works
    };
    console.log("Using gas options:", gasOptions);


    // --- Scenario 1: Product 1 (DEMO_PROD_001) Listing and Purchase Intent by Buyer1 ---
    const product1UniqueId = "DEMO_PROD_001";
    const product1Info = productDetails.find(p => p.uniqueProductID === product1UniqueId);
    if (!product1Info) {
        console.error(`Product with uniqueID ${product1UniqueId} not found in demo_context.json. Ensure script 02 creates it.`);
        process.exit(1);
    }
    // Update productInfo with CLI arguments for consistency and for use in transactions
    product1Info.tokenId = tokenIdArg1;
    // product1Info.initialCID = ipfsCIDArg1; // No longer storing this directly in product1Info this way
    
    if (product1Info.currentOwnerAddress.toLowerCase() !== manufacturerAcc.address.toLowerCase()) {
        console.error(`Product 1 owner in context is ${product1Info.currentOwnerAddress}, not manufacturer ${manufacturerAcc.address}. Check script 02 output.`);
        process.exit(1);
    }

    const tokenId1 = product1Info.tokenId; 
    const price1 = ethers.parseEther("0.1"); 
    const collateral1 = price1; 
    const cidForVerification1 = ipfsCIDArg1; // Use the IPFS CID from env var

    if (!cidForVerification1 || cidForVerification1.toLowerCase() === 'null' || cidForVerification1.trim() === '') {
        console.error(`Error: IPFS CID for Product 1 (Token ID: ${tokenId1}) is missing or invalid from command line arguments.`);
        process.exit(1);
    }

    console.log(`\\n--- Product 1 (Unique ID: ${product1UniqueId}, Token ID: ${tokenId1}): Listing and Collateral Deposit by Buyer 1 ---`);
    console.log(`  Manufacturer (${manufacturerAcc.address}) lists Token ID ${tokenId1} for ${ethers.formatEther(price1)} ETH.`);
    let sellTx1 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId1, price1, gasOptions);
    await sellTx1.wait(1);
    console.log("    Product 1 listed for sale by manufacturer.");

    product1Info.price = price1.toString();
    product1Info.sellerAddress = manufacturerAcc.address;
    product1Info.productStatus = "ListedForSale";

    console.log(`  Buyer 1 (${buyer1Acc.address}) initiates purchase for Token ID ${tokenId1} using CID for verification: ${cidForVerification1}.`);
    let initiateTx1 = await supplyChainNFT.connect(buyer1Acc).initiatePurchase(tokenId1, cidForVerification1, gasOptions);
    await initiateTx1.wait(1);
    console.log(`    Purchase initiated by Buyer 1. Product status should be AwaitingCollateral.`);
    
    product1Info.productStatus = "AwaitingCollateral";
    product1Info.pendingBuyerAddress = buyer1Acc.address;

    console.log(`  Buyer 1 (${buyer1Acc.address}) deposits collateral of ${ethers.formatEther(collateral1)} ETH for Token ID ${tokenId1}.`);
    let depositTx1 = await supplyChainNFT.connect(buyer1Acc).depositPurchaseCollateral(tokenId1, { ...gasOptions, value: collateral1 });
    await depositTx1.wait(1);
    console.log(`    Collateral deposited by Buyer 1. NFT remains with seller (${manufacturerAcc.address}).`);
    console.log("    ACTION: backendListener.js should detect CollateralDepositedForPurchase event for Product 1.");

    product1Info.collateralAmount = collateral1.toString();
    product1Info.productStatus = "CollateralDeposited";

    // --- Scenario 2: Product 2 (DEMO_PROD_002) & Product 3 (DEMO_PROD_003) Listing and Purchase Intent by Buyer2 (Retailer) ---
    
    // Product 2 (DEMO_PROD_002)
    const product2UniqueId = "DEMO_PROD_002";
    const product2Info = productDetails.find(p => p.uniqueProductID === product2UniqueId);
    if (!product2Info) {
        console.error(`Product with uniqueID ${product2UniqueId} not found in demo_context.json. Ensure script 02 creates it.`);
        process.exit(1);
    }
    // Update productInfo with CLI arguments
    product2Info.tokenId = tokenIdArg2;
    // product2Info.initialCID = ipfsCIDArg2;

    if (product2Info.currentOwnerAddress.toLowerCase() !== manufacturerAcc.address.toLowerCase()) {
        console.error(`Product 2 owner in context is ${product2Info.currentOwnerAddress}, not manufacturer ${manufacturerAcc.address}. Check script 02 output.`);
        process.exit(1);
    }

    const tokenId2 = product2Info.tokenId; 
    const price2 = ethers.parseEther("0.2"); 
    const collateral2 = price2; 
    const cidForVerification2 = ipfsCIDArg2; // Use the IPFS CID from env var

    if (!cidForVerification2 || cidForVerification2.toLowerCase() === 'null' || cidForVerification2.trim() === '') {
        console.error(`Error: IPFS CID for Product 2 (Token ID: ${tokenId2}) is missing or invalid from command line arguments.`);
        process.exit(1);
    }

    console.log(`\\n--- Product 2 (Unique ID: ${product2UniqueId}, Token ID: ${tokenId2}): Listing and Collateral Deposit by Buyer 2 (Retailer) ---`);
    console.log(`  Manufacturer (${manufacturerAcc.address}) lists Token ID ${tokenId2} for ${ethers.formatEther(price2)} ETH.`);
    let sellTx2 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId2, price2, gasOptions);
    await sellTx2.wait(1);
    console.log("    Product 2 listed for sale by manufacturer.");

    product2Info.price = price2.toString();
    product2Info.sellerAddress = manufacturerAcc.address;
    product2Info.productStatus = "ListedForSale";

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) initiates purchase for Token ID ${tokenId2} using CID for verification: ${cidForVerification2}.`);
    let initiateTx2 = await supplyChainNFT.connect(retailerAcc).initiatePurchase(tokenId2, cidForVerification2, gasOptions);
    await initiateTx2.wait(1);
    console.log(`    Purchase initiated by Retailer/Buyer 2 for Product 2. Product status should be AwaitingCollateral.`);

    product2Info.productStatus = "AwaitingCollateral";
    product2Info.pendingBuyerAddress = retailerAcc.address;

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) deposits collateral of ${ethers.formatEther(collateral2)} ETH for Token ID ${tokenId2}.`);
    let depositTx2 = await supplyChainNFT.connect(retailerAcc).depositPurchaseCollateral(tokenId2, { ...gasOptions, value: collateral2 });
    await depositTx2.wait(1);
    console.log(`    Collateral deposited by Retailer/Buyer 2 for Product 2. NFT remains with seller (${manufacturerAcc.address}).`);
    console.log("    ACTION: backendListener.js should detect CollateralDepositedForPurchase event for Product 2.");
    
    product2Info.collateralAmount = collateral2.toString();
    product2Info.productStatus = "CollateralDeposited";

    // Product 3 (DEMO_PROD_003)
    const product3UniqueId = "DEMO_PROD_003"; // Assuming script 02 creates this product entry
    const product3Info = productDetails.find(p => p.uniqueProductID === product3UniqueId);
    if (!product3Info) {
        console.error(`Product with uniqueID ${product3UniqueId} not found in demo_context.json. Ensure script 02 creates it.`);
        process.exit(1);
    }
    // Update productInfo with CLI arguments
    product3Info.tokenId = tokenIdArg3;
    // product3Info.initialCID = ipfsCIDArg3;

    if (product3Info.currentOwnerAddress.toLowerCase() !== manufacturerAcc.address.toLowerCase()) {
        console.error(`Product 3 owner in context is ${product3Info.currentOwnerAddress}, not manufacturer ${manufacturerAcc.address}. Check script 02 output.`);
        process.exit(1);
    }

    const tokenId3 = product3Info.tokenId; 
    const price3 = ethers.parseEther("0.3"); 
    const collateral3 = price3; 
    const cidForVerification3 = ipfsCIDArg3; // Use the IPFS CID from env var

    if (!cidForVerification3 || cidForVerification3.toLowerCase() === 'null' || cidForVerification3.trim() === '') {
        console.error(`Error: IPFS CID for Product 3 (Token ID: ${tokenId3}) is missing or invalid from command line arguments.`);
        process.exit(1);
    }

    console.log(`\\n--- Product 3 (Unique ID: ${product3UniqueId}, Token ID: ${tokenId3}): Listing and Collateral Deposit by Buyer 2 (Retailer) ---`);
    console.log(`  Manufacturer (${manufacturerAcc.address}) lists Token ID ${tokenId3} for ${ethers.formatEther(price3)} ETH.`);
    let sellTx3 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId3, price3, gasOptions);
    await sellTx3.wait(1);
    console.log("    Product 3 listed for sale by manufacturer.");

    product3Info.price = price3.toString();
    product3Info.sellerAddress = manufacturerAcc.address;
    product3Info.productStatus = "ListedForSale";

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) initiates purchase for Token ID ${tokenId3} using CID for verification: ${cidForVerification3}.`);
    let initiateTx3 = await supplyChainNFT.connect(retailerAcc).initiatePurchase(tokenId3, cidForVerification3, gasOptions);
    await initiateTx3.wait(1);
    console.log(`    Purchase initiated by Retailer/Buyer 2 for Product 3. Product status should be AwaitingCollateral.`);

    product3Info.productStatus = "AwaitingCollateral";
    product3Info.pendingBuyerAddress = retailerAcc.address; // Same buyer for product 2 and 3 in this scenario

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) deposits collateral of ${ethers.formatEther(collateral3)} ETH for Token ID ${tokenId3}.`);
    let depositTx3 = await supplyChainNFT.connect(retailerAcc).depositPurchaseCollateral(tokenId3, { ...gasOptions, value: collateral3 });
    await depositTx3.wait(1);
    console.log(`    Collateral deposited by Retailer/Buyer 2 for Product 3. NFT remains with seller (${manufacturerAcc.address}).`);
    console.log("    ACTION: backendListener.js should detect CollateralDepositedForPurchase event for Product 3.");
    
    product3Info.collateralAmount = collateral3.toString();
    product3Info.productStatus = "CollateralDeposited";
    
    // Save context
    saveDemoContext(context);

    console.log("\\n--- 03: Marketplace Listing and Collateral Deposit Scenario Complete ---");
    console.log("Key outcomes:");
    // Ensure manufacturerAcc.address is used for current owner as NFT not transferred yet
    console.log(`  - Product 1 (ID: ${tokenId1}, UniqueID: ${product1UniqueId}): Listed by ${product1Info.sellerAddress}, collateral deposited by ${product1Info.pendingBuyerAddress}. Current Owner (in contract): ${manufacturerAcc.address}`);
    console.log(`  - Product 2 (ID: ${tokenId2}, UniqueID: ${product2UniqueId}): Listed by ${product2Info.sellerAddress}, collateral deposited by ${product2Info.pendingBuyerAddress}. Current Owner (in contract): ${manufacturerAcc.address}`);
    console.log(`  - Product 3 (ID: ${tokenId3}, UniqueID: ${product3UniqueId}): Listed by ${product3Info.sellerAddress}, collateral deposited by ${product3Info.pendingBuyerAddress}. Current Owner (in contract): ${manufacturerAcc.address}`);
    console.log("  NFTs have NOT been transferred. Payments have NOT been released to sellers.");
    console.log("  Next step: 04_scenario_transport_and_ipfs.cjs for transport and purchase finalization.");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

