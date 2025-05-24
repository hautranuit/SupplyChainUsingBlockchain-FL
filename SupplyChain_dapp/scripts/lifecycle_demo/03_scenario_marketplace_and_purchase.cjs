const { ethers } = require("hardhat");
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

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
    console.log("--- Starting 03: Marketplace Listing and Collateral Deposit (with Real CID Update) ---");

    // Read arguments from environment variables set by the Hardhat task
    const tokenIdArg1 = process.env.SCENARIO03_TOKENID1;
    const ipfsCIDArg1 = process.env.SCENARIO03_CID1; // Real CID for Token 1
    const tokenIdArg2 = process.env.SCENARIO03_TOKENID2;
    const ipfsCIDArg2 = process.env.SCENARIO03_CID2; // Real CID for Token 2
    const tokenIdArg3 = process.env.SCENARIO03_TOKENID3;
    const ipfsCIDArg3 = process.env.SCENARIO03_CID3; // Real CID for Token 3

    if (!tokenIdArg1 || !ipfsCIDArg1 || !tokenIdArg2 || !ipfsCIDArg2 || !tokenIdArg3 || !ipfsCIDArg3) {
        console.error("Error: Missing one or more required environment variables for token IDs and REAL IPFS CIDs.");
        console.error("Ensure SCENARIO03_TOKENID1, SCENARIO03_CID1, etc., are set when calling the 'run-scenario03' Hardhat task.");
        process.exit(1);
    }

    console.log("Using parameters from environment variables (via Hardhat task):");
    console.log(`  Product 1 (DEMO_PROD_001): Token ID = ${tokenIdArg1}, Real IPFS CID = ${ipfsCIDArg1}`);
    console.log(`  Product 2 (DEMO_PROD_002): Token ID = ${tokenIdArg2}, Real IPFS CID = ${ipfsCIDArg2}`);
    console.log(`  Product 3 (DEMO_PROD_003): Token ID = ${tokenIdArg3}, Real IPFS CID = ${ipfsCIDArg3}`);

    // Use readAndUpdateContext to load and prepare for saving
    let finalContext = readAndUpdateContext(currentContext => {
        const contractAddress = currentContext.contractAddress;
        // Check if using productDetails array or products object
        let productsSource = currentContext.products || currentContext.productDetails;
        if (!contractAddress || !productsSource) {
            console.error("Error: Invalid context. Ensure contractAddress and products/productDetails are present from previous scripts.");
            // Return unmodified context to avoid writing partial/error state
            return currentContext; 
        }
        console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

        // Store the real CIDs received from args into the context
        // Assuming productsSource is an object keyed by tokenId (as created by script 02_no_cid)
        if (productsSource[tokenIdArg1]) {
            productsSource[tokenIdArg1].nftReference = ipfsCIDArg1; // Update with real CID
            console.log(`  Updated context for Token ID ${tokenIdArg1} with real CID: ${ipfsCIDArg1}`);
        } else {
            console.warn(`  WARN: Token ID ${tokenIdArg1} not found in context. Cannot update CID.`);
        }
        if (productsSource[tokenIdArg2]) {
            productsSource[tokenIdArg2].nftReference = ipfsCIDArg2; // Update with real CID
            console.log(`  Updated context for Token ID ${tokenIdArg2} with real CID: ${ipfsCIDArg2}`);
        } else {
            console.warn(`  WARN: Token ID ${tokenIdArg2} not found in context. Cannot update CID.`);
        }
        if (productsSource[tokenIdArg3]) {
            productsSource[tokenIdArg3].nftReference = ipfsCIDArg3; // Update with real CID
            console.log(`  Updated context for Token ID ${tokenIdArg3} with real CID: ${ipfsCIDArg3}`);
        } else {
            console.warn(`  WARN: Token ID ${tokenIdArg3} not found in context. Cannot update CID.`);
        }
        
        // Return the modified context so readAndUpdateContext saves it
        return currentContext; 
    });

    // Re-read the updated context for subsequent operations in this script
    const context = finalContext; 
    const contractAddress = context.contractAddress;
    const products = context.products; // Use the updated products object

    if (!contractAddress || !products) {
        console.error("Error: Failed to load or update context correctly.");
        process.exit(1);
    }

    const signers = await ethers.getSigners();
    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }

    const manufacturerAcc = signers.find(s => s.address.toLowerCase() === Object.keys(context.nodes).find(k => context.nodes[k].name === "Manufacturer"));
    const buyer1Acc = signers.find(s => s.address.toLowerCase() === Object.keys(context.nodes).find(k => context.nodes[k].name === "Buyer/Customer"));
    const retailerAcc = signers.find(s => s.address.toLowerCase() === Object.keys(context.nodes).find(k => context.nodes[k].name === "Retailer")); // Acting as Buyer 2

    if (!manufacturerAcc || !buyer1Acc || !retailerAcc) {
        console.error("Error: Could not find one or more required accounts (Manufacturer, Buyer1, Retailer) in context.");
        process.exit(1);
    }

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress);
    console.log("Connected to contract.");

    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'),
        maxFeePerGas: ethers.parseUnits('40', 'gwei')
    };
    // console.log("Using gas options:", gasOptions);

    // --- Scenario 1: Product 1 (DEMO_PROD_001) Listing and Purchase Intent by Buyer1 ---
    const product1Info = products[tokenIdArg1];
    if (!product1Info) {
        console.error(`Product with Token ID ${tokenIdArg1} not found in updated context.`);
        process.exit(1);
    }
    if (product1Info.currentOwnerAddress.toLowerCase() !== manufacturerAcc.address.toLowerCase()) {
        console.error(`Product 1 owner in context is ${product1Info.currentOwnerAddress}, not manufacturer ${manufacturerAcc.address}. Check script 02 output.`);
        process.exit(1);
    }

    const tokenId1 = product1Info.tokenId;
    const price1 = ethers.parseEther("0.1");
    const collateral1 = price1;
    const cidForVerification1 = product1Info.nftReference; // Use the REAL CID updated in context

    if (!cidForVerification1 || cidForVerification1.trim() === '') {
        console.error(`Error: Real IPFS CID for Product 1 (Token ID: ${tokenId1}) is missing in context after update.`);
        process.exit(1);
    }

    console.log(`\n--- Product 1 (Token ID: ${tokenId1}): Listing and Collateral Deposit by Buyer 1 ---`);
    console.log(`  Manufacturer (${manufacturerAcc.address}) lists Token ID ${tokenId1} for ${ethers.formatEther(price1)} ETH.`);
    let sellTx1 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId1, price1, gasOptions);
    let sellReceipt1 = await sellTx1.wait(1);
    let sellBlock1 = await ethers.provider.getBlock(sellReceipt1.blockNumber);
    console.log("    Product 1 listed for sale by manufacturer.");

    // Update context after listing
    readAndUpdateContext(ctx => {
        if (ctx.products[tokenId1]) {
            ctx.products[tokenId1].price = price1.toString();
            ctx.products[tokenId1].sellerAddress = manufacturerAcc.address.toLowerCase();
            ctx.products[tokenId1].status = "ListedForSale";
            ctx.products[tokenId1].lastUpdateTimestamp = sellBlock1.timestamp;
            // Update manufacturer interactions
            if (ctx.nodes[manufacturerAcc.address.toLowerCase()]) {
                if (!ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions) ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions = [];
                ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({
                    type: "ListProduct", tokenId: tokenId1, price: ethers.formatEther(price1) + " ETH", 
                    timestamp: sellBlock1.timestamp, txHash: sellReceipt1.transactionHash
                });
            }
        }
        return ctx;
    });

    console.log(`  Buyer 1 (${buyer1Acc.address}) initiates purchase for Token ID ${tokenId1} using REAL CID for verification: ${cidForVerification1}.`);
    // *** IMPORTANT: The contract's initiatePurchase likely still needs the CID that was stored on-chain.
    // If storeInitialCID was never called (as in script 02_no_cid), this initiatePurchase might fail
    // unless the contract logic was changed to NOT require a pre-stored CID for initiation.
    // Assuming for now initiatePurchase DOES NOT require a pre-stored CID, or that a separate mechanism stored the real CID.
    let initiateTx1 = await supplyChainNFT.connect(buyer1Acc).initiatePurchase(tokenId1, cidForVerification1, gasOptions);
    let initiateReceipt1 = await initiateTx1.wait(1);
    let initiateBlock1 = await ethers.provider.getBlock(initiateReceipt1.blockNumber);
    console.log(`    Purchase initiated by Buyer 1. Product status should be AwaitingCollateral.`);

    // Update context after initiation
     readAndUpdateContext(ctx => {
        if (ctx.products[tokenId1]) {
            ctx.products[tokenId1].status = "AwaitingCollateral";
            ctx.products[tokenId1].pendingBuyerAddress = buyer1Acc.address.toLowerCase();
            ctx.products[tokenId1].lastUpdateTimestamp = initiateBlock1.timestamp;
             // Update buyer interactions
            if (ctx.nodes[buyer1Acc.address.toLowerCase()]) {
                if (!ctx.nodes[buyer1Acc.address.toLowerCase()].interactions) ctx.nodes[buyer1Acc.address.toLowerCase()].interactions = [];
                ctx.nodes[buyer1Acc.address.toLowerCase()].interactions.push({
                    type: "InitiatePurchase", tokenId: tokenId1, cidUsed: cidForVerification1,
                    timestamp: initiateBlock1.timestamp, txHash: initiateReceipt1.transactionHash
                });
            }
        }
        return ctx;
    });

    console.log(`  Buyer 1 (${buyer1Acc.address}) deposits collateral of ${ethers.formatEther(collateral1)} ETH for Token ID ${tokenId1}.`);
    let depositTx1 = await supplyChainNFT.connect(buyer1Acc).depositPurchaseCollateral(tokenId1, { ...gasOptions, value: collateral1 });
    let depositReceipt1 = await depositTx1.wait(1);
    let depositBlock1 = await ethers.provider.getBlock(depositReceipt1.blockNumber);
    console.log(`    Collateral deposited by Buyer 1. NFT remains with seller (${manufacturerAcc.address}).`);

    // Update context after deposit
    readAndUpdateContext(ctx => {
        if (ctx.products[tokenId1]) {
            ctx.products[tokenId1].collateralAmount = collateral1.toString();
            ctx.products[tokenId1].status = "CollateralDeposited";
            ctx.products[tokenId1].lastUpdateTimestamp = depositBlock1.timestamp;
            // Update buyer interactions
            if (ctx.nodes[buyer1Acc.address.toLowerCase()]) {
                 if (!ctx.nodes[buyer1Acc.address.toLowerCase()].interactions) ctx.nodes[buyer1Acc.address.toLowerCase()].interactions = [];
                 ctx.nodes[buyer1Acc.address.toLowerCase()].interactions.push({
                    type: "DepositCollateral", tokenId: tokenId1, amount: ethers.formatEther(collateral1) + " ETH",
                    timestamp: depositBlock1.timestamp, txHash: depositReceipt1.transactionHash
                });
            }
        }
        return ctx;
    });

    // --- Scenario 2 & 3: Product 2 & 3 Listing and Purchase Intent by Buyer2 (Retailer) ---
    // Similar logic for Product 2
    const product2Info = products[tokenIdArg2];
    if (!product2Info) { console.error(`Product 2 (ID: ${tokenIdArg2}) not found.`); process.exit(1); }
    const tokenId2 = product2Info.tokenId;
    const price2 = ethers.parseEther("0.2");
    const collateral2 = price2;
    const cidForVerification2 = product2Info.nftReference; // Use REAL CID from context
    if (!cidForVerification2 || cidForVerification2.trim() === '') { console.error(`Real CID missing for Product 2 (ID: ${tokenId2})`); process.exit(1); }

    console.log(`\n--- Product 2 (Token ID: ${tokenId2}): Listing and Collateral Deposit by Buyer 2 (Retailer) ---`);
    console.log(`  Manufacturer lists Token ID ${tokenId2} for ${ethers.formatEther(price2)} ETH.`);
    let sellTx2 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId2, price2, gasOptions);
    let sellReceipt2 = await sellTx2.wait(1); let sellBlock2 = await ethers.provider.getBlock(sellReceipt2.blockNumber);
    console.log("    Product 2 listed.");
    readAndUpdateContext(ctx => { /* Update context for product 2 listing */ 
        if(ctx.products[tokenId2]){ ctx.products[tokenId2].price = price2.toString(); ctx.products[tokenId2].sellerAddress = manufacturerAcc.address.toLowerCase(); ctx.products[tokenId2].status = "ListedForSale"; ctx.products[tokenId2].lastUpdateTimestamp = sellBlock2.timestamp; }
        if(ctx.nodes[manufacturerAcc.address.toLowerCase()]){ if(!ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions) ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({type:"ListProduct", tokenId:tokenId2, price: ethers.formatEther(price2)+" ETH", timestamp: sellBlock2.timestamp, txHash: sellReceipt2.transactionHash});}
        return ctx; 
    });

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) initiates purchase for Token ID ${tokenId2} using REAL CID: ${cidForVerification2}.`);
    let initiateTx2 = await supplyChainNFT.connect(retailerAcc).initiatePurchase(tokenId2, cidForVerification2, gasOptions);
    let initiateReceipt2 = await initiateTx2.wait(1); let initiateBlock2 = await ethers.provider.getBlock(initiateReceipt2.blockNumber);
    console.log(`    Purchase initiated for Product 2.`);
    readAndUpdateContext(ctx => { /* Update context for product 2 initiation */ 
        if(ctx.products[tokenId2]){ ctx.products[tokenId2].status = "AwaitingCollateral"; ctx.products[tokenId2].pendingBuyerAddress = retailerAcc.address.toLowerCase(); ctx.products[tokenId2].lastUpdateTimestamp = initiateBlock2.timestamp; }
        if(ctx.nodes[retailerAcc.address.toLowerCase()]){ if(!ctx.nodes[retailerAcc.address.toLowerCase()].interactions) ctx.nodes[retailerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[retailerAcc.address.toLowerCase()].interactions.push({type:"InitiatePurchase", tokenId:tokenId2, cidUsed: cidForVerification2, timestamp: initiateBlock2.timestamp, txHash: initiateReceipt2.transactionHash});}
        return ctx; 
    });

    console.log(`  Retailer/Buyer 2 deposits collateral of ${ethers.formatEther(collateral2)} ETH for Token ID ${tokenId2}.`);
    let depositTx2 = await supplyChainNFT.connect(retailerAcc).depositPurchaseCollateral(tokenId2, { ...gasOptions, value: collateral2 });
    let depositReceipt2 = await depositTx2.wait(1); let depositBlock2 = await ethers.provider.getBlock(depositReceipt2.blockNumber);
    console.log(`    Collateral deposited for Product 2.`);
    readAndUpdateContext(ctx => { /* Update context for product 2 deposit */ 
        if(ctx.products[tokenId2]){ ctx.products[tokenId2].collateralAmount = collateral2.toString(); ctx.products[tokenId2].status = "CollateralDeposited"; ctx.products[tokenId2].lastUpdateTimestamp = depositBlock2.timestamp; }
        if(ctx.nodes[retailerAcc.address.toLowerCase()]){ if(!ctx.nodes[retailerAcc.address.toLowerCase()].interactions) ctx.nodes[retailerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[retailerAcc.address.toLowerCase()].interactions.push({type:"DepositCollateral", tokenId:tokenId2, amount: ethers.formatEther(collateral2)+" ETH", timestamp: depositBlock2.timestamp, txHash: depositReceipt2.transactionHash});}
        return ctx; 
    });

    // Similar logic for Product 3
    const product3Info = products[tokenIdArg3];
    if (!product3Info) { console.error(`Product 3 (ID: ${tokenIdArg3}) not found.`); process.exit(1); }
    const tokenId3 = product3Info.tokenId;
    const price3 = ethers.parseEther("0.3");
    const collateral3 = price3;
    const cidForVerification3 = product3Info.nftReference; // Use REAL CID from context
    if (!cidForVerification3 || cidForVerification3.trim() === '') { console.error(`Real CID missing for Product 3 (ID: ${tokenId3})`); process.exit(1); }

    console.log(`\n--- Product 3 (Token ID: ${tokenId3}): Listing and Collateral Deposit by Buyer 2 (Retailer) ---`);
    console.log(`  Manufacturer lists Token ID ${tokenId3} for ${ethers.formatEther(price3)} ETH.`);
    let sellTx3 = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId3, price3, gasOptions);
    let sellReceipt3 = await sellTx3.wait(1); let sellBlock3 = await ethers.provider.getBlock(sellReceipt3.blockNumber);
    console.log("    Product 3 listed.");
    readAndUpdateContext(ctx => { /* Update context for product 3 listing */ 
        if(ctx.products[tokenId3]){ ctx.products[tokenId3].price = price3.toString(); ctx.products[tokenId3].sellerAddress = manufacturerAcc.address.toLowerCase(); ctx.products[tokenId3].status = "ListedForSale"; ctx.products[tokenId3].lastUpdateTimestamp = sellBlock3.timestamp; }
        if(ctx.nodes[manufacturerAcc.address.toLowerCase()]){ if(!ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions) ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({type:"ListProduct", tokenId:tokenId3, price: ethers.formatEther(price3)+" ETH", timestamp: sellBlock3.timestamp, txHash: sellReceipt3.transactionHash});}
        return ctx; 
    });

    console.log(`  Retailer/Buyer 2 (${retailerAcc.address}) initiates purchase for Token ID ${tokenId3} using REAL CID: ${cidForVerification3}.`);
    let initiateTx3 = await supplyChainNFT.connect(retailerAcc).initiatePurchase(tokenId3, cidForVerification3, gasOptions);
    let initiateReceipt3 = await initiateTx3.wait(1); let initiateBlock3 = await ethers.provider.getBlock(initiateReceipt3.blockNumber);
    console.log(`    Purchase initiated for Product 3.`);
    readAndUpdateContext(ctx => { /* Update context for product 3 initiation */ 
        if(ctx.products[tokenId3]){ ctx.products[tokenId3].status = "AwaitingCollateral"; ctx.products[tokenId3].pendingBuyerAddress = retailerAcc.address.toLowerCase(); ctx.products[tokenId3].lastUpdateTimestamp = initiateBlock3.timestamp; }
        if(ctx.nodes[retailerAcc.address.toLowerCase()]){ if(!ctx.nodes[retailerAcc.address.toLowerCase()].interactions) ctx.nodes[retailerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[retailerAcc.address.toLowerCase()].interactions.push({type:"InitiatePurchase", tokenId:tokenId3, cidUsed: cidForVerification3, timestamp: initiateBlock3.timestamp, txHash: initiateReceipt3.transactionHash});}
        return ctx; 
    });

    console.log(`  Retailer/Buyer 2 deposits collateral of ${ethers.formatEther(collateral3)} ETH for Token ID ${tokenId3}.`);
    let depositTx3 = await supplyChainNFT.connect(retailerAcc).depositPurchaseCollateral(tokenId3, { ...gasOptions, value: collateral3 });
    let depositReceipt3 = await depositTx3.wait(1); let depositBlock3 = await ethers.provider.getBlock(depositReceipt3.blockNumber);
    console.log(`    Collateral deposited for Product 3.`);
    readAndUpdateContext(ctx => { /* Update context for product 3 deposit */ 
        if(ctx.products[tokenId3]){ ctx.products[tokenId3].collateralAmount = collateral3.toString(); ctx.products[tokenId3].status = "CollateralDeposited"; ctx.products[tokenId3].lastUpdateTimestamp = depositBlock3.timestamp; }
        if(ctx.nodes[retailerAcc.address.toLowerCase()]){ if(!ctx.nodes[retailerAcc.address.toLowerCase()].interactions) ctx.nodes[retailerAcc.address.toLowerCase()].interactions=[]; ctx.nodes[retailerAcc.address.toLowerCase()].interactions.push({type:"DepositCollateral", tokenId:tokenId3, amount: ethers.formatEther(collateral3)+" ETH", timestamp: depositBlock3.timestamp, txHash: depositReceipt3.transactionHash});}
        return ctx; 
    });

    console.log("\n--- 03: Marketplace Listing and Collateral Deposit Scenario Complete --- (Real CIDs Updated in Context)");
    console.log("Key outcomes:");
    console.log(`  - Product 1 (ID: ${tokenId1}): Listed, collateral deposited. Real CID: ${cidForVerification1} updated in context.`);
    console.log(`  - Product 2 (ID: ${tokenId2}): Listed, collateral deposited. Real CID: ${cidForVerification2} updated in context.`);
    console.log(`  - Product 3 (ID: ${tokenId3}): Listed, collateral deposited. Real CID: ${cidForVerification3} updated in context.`);
    console.log("  Next step: 04_scenario_transport_and_ipfs.cjs");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

