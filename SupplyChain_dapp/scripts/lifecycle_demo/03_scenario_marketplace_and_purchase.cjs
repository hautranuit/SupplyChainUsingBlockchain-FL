const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read demo_context.json
function getDemoContext() {
    const demoContextPath = path.join(__dirname, "demo_context.json");
    if (!fs.existsSync(demoContextPath)) {
        console.error(`Error: demo_context.json not found at ${demoContextPath}`);
        console.error("Please run 02_scenario_product_creation.js first.");
        process.exit(1);
    }
    const contextContent = fs.readFileSync(demoContextPath, "utf8");
    return JSON.parse(contextContent);
}

async function main() {
    console.log("--- Starting 03: Marketplace and Purchase Scenarios ---");

    const context = getDemoContext();
    const contractAddress = context.contractAddress;
    const mintedTokenIds = context.tokenIds;
    const productDetails = context.productDetails;

    if (!contractAddress || !mintedTokenIds || mintedTokenIds.length < 3 || !productDetails) {
        console.error("Error: Invalid context. Ensure contractAddress and at least 3 tokenIds are present.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    // Script 01 configured these signers:
    // deployer = signers[0];
    // manufacturerAcc = signers[1];
    // transporter1Acc = signers[2];
    // transporter2Acc = signers[3];
    // transporter3Acc = signers[4];
    // retailerAcc = signers[5];
    // buyerAcc = signers[6];
    // arbitratorAcc = signers[7];

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers as configured in 01_deploy_and_configure.js.");
        process.exit(1);
    }

    const deployer = signers[0];
    const manufacturerAcc = signers[1];
    const buyer1Acc = signers[6]; // Using the configured buyer
    const buyer2Acc = signers[5]; // Using retailer as another buyer for variety

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log("Connected to contract.");

    // --- Scenario 1: List Product 1 for Sale & Buyer 1 Purchases ---
    const tokenId1 = mintedTokenIds[0];
    const product1Details = productDetails.find(p => p.tokenId === tokenId1);
    if (!product1Details) {
        console.error(`Could not find details for tokenId1: ${tokenId1}`);
        process.exit(1);
    }
    const price1 = ethers.parseEther("0.1"); // 0.1 ETH
    console.log(`\n--- Scenario 1: Product ${product1Details.uniqueProductID} (ID: ${tokenId1}) ---`);
    console.log(`Manufacturer (${manufacturerAcc.address}) listing Token ID ${tokenId1} for sale at ${ethers.formatEther(price1)} ETH...`);
    let tx = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId1, price1);
    let receipt = await tx.wait(1);
    console.log(`  Product listed. Gas Used: ${receipt.gasUsed.toString()}`);

    console.log(`Buyer 1 (${buyer1Acc.address}) initiating purchase for Token ID ${tokenId1}...`);
    // Buyer needs to provide the current CID for authenticity verification
    // Using the placeholder from minting script for demo purposes
    const currentCID1 = product1Details.nftReference; // Or a more specific CID if updated
    tx = await supplyChainNFT.connect(buyer1Acc).initiatePurchase(tokenId1, currentCID1);
    receipt = await tx.wait(1);
    console.log(`  Purchase initiated. Gas Used: ${receipt.gasUsed.toString()}`);

    console.log(`Buyer 1 (${buyer1Acc.address}) depositing collateral for Token ID ${tokenId1}...`);
    tx = await supplyChainNFT.connect(buyer1Acc).depositPurchaseCollateral(tokenId1, { value: price1 });
    receipt = await tx.wait(1);
    console.log(`  Collateral deposited. Gas Used: ${receipt.gasUsed.toString()}`);
    // At this point, ownership is transferred. We can verify.
    const ownerOfToken1 = await supplyChainNFT.ownerOf(tokenId1);
    console.log(`  Current owner of Token ID ${tokenId1}: ${ownerOfToken1} (Expected: ${buyer1Acc.address})`);
    if (ownerOfToken1 !== buyer1Acc.address) {
        console.error("  ERROR: Ownership transfer failed for Token ID 1!");
    }

    // --- Scenario 2: List Product 2 for Sale & Buyer 2 Purchases ---
    const tokenId2 = mintedTokenIds[1];
    const product2Details = productDetails.find(p => p.tokenId === tokenId2);
    if (!product2Details) {
        console.error(`Could not find details for tokenId2: ${tokenId2}`);
        process.exit(1);
    }
    const price2 = ethers.parseEther("0.05"); // 0.05 ETH
    console.log(`\n--- Scenario 2: Product ${product2Details.uniqueProductID} (ID: ${tokenId2}) ---`);
    console.log(`Manufacturer (${manufacturerAcc.address}) listing Token ID ${tokenId2} for sale at ${ethers.formatEther(price2)} ETH...`);
    tx = await supplyChainNFT.connect(manufacturerAcc).sellProduct(tokenId2, price2);
    receipt = await tx.wait(1);
    console.log(`  Product listed. Gas Used: ${receipt.gasUsed.toString()}`);

    console.log(`Buyer 2 (${buyer2Acc.address}) initiating purchase for Token ID ${tokenId2}...`);
    const currentCID2 = product2Details.nftReference;
    tx = await supplyChainNFT.connect(buyer2Acc).initiatePurchase(tokenId2, currentCID2);
    receipt = await tx.wait(1);
    console.log(`  Purchase initiated. Gas Used: ${receipt.gasUsed.toString()}`);

    console.log(`Buyer 2 (${buyer2Acc.address}) depositing collateral for Token ID ${tokenId2}...`);
    tx = await supplyChainNFT.connect(buyer2Acc).depositPurchaseCollateral(tokenId2, { value: price2 });
    receipt = await tx.wait(1);
    console.log(`  Collateral deposited. Gas Used: ${receipt.gasUsed.toString()}`);
    const ownerOfToken2 = await supplyChainNFT.ownerOf(tokenId2);
    console.log(`  Current owner of Token ID ${tokenId2}: ${ownerOfToken2} (Expected: ${buyer2Acc.address})`);
    if (ownerOfToken2 !== buyer2Acc.address) {
        console.error("  ERROR: Ownership transfer failed for Token ID 2!");
    }

    // --- Scenario 3: Direct Sale and Transfer for Product 3 ---
    const tokenId3 = mintedTokenIds[2];
    const product3Details = productDetails.find(p => p.tokenId === tokenId3);
    if (!product3Details) {
        console.error(`Could not find details for tokenId3: ${tokenId3}`);
        process.exit(1);
    }
    const price3 = ethers.parseEther("0.2"); // 0.2 ETH
    const directBuyerAcc = buyer1Acc; // Re-using buyer1 for this direct sale
    console.log(`\n--- Scenario 3: Direct Sale of Product ${product3Details.uniqueProductID} (ID: ${tokenId3}) ---`);
    console.log(`Manufacturer (${manufacturerAcc.address}) directly selling Token ID ${tokenId3} to ${directBuyerAcc.address} for ${ethers.formatEther(price3)} ETH...`);
    const currentCID3 = product3Details.nftReference;
    tx = await supplyChainNFT.connect(manufacturerAcc).sellAndTransferProduct(tokenId3, price3, directBuyerAcc.address, currentCID3);
    receipt = await tx.wait(1);
    console.log(`  Direct sale and transfer completed. Gas Used: ${receipt.gasUsed.toString()}`);
    const ownerOfToken3 = await supplyChainNFT.ownerOf(tokenId3);
    console.log(`  Current owner of Token ID ${tokenId3}: ${ownerOfToken3} (Expected: ${directBuyerAcc.address})`);
    if (ownerOfToken3 !== directBuyerAcc.address) {
        console.error("  ERROR: Ownership transfer failed for Token ID 3 (Direct Sale)!");
    }

    // Update context with new owners for transport script
    const updatedProductDetails = context.productDetails.map(p => {
        if (p.tokenId === tokenId1) return { ...p, currentOwnerAddress: buyer1Acc.address, purchasePrice: price1.toString() };
        if (p.tokenId === tokenId2) return { ...p, currentOwnerAddress: buyer2Acc.address, purchasePrice: price2.toString() };
        if (p.tokenId === tokenId3) return { ...p, currentOwnerAddress: directBuyerAcc.address, purchasePrice: price3.toString() }; // directBuyerAcc is buyer1Acc
        return p;
    });

    const newDemoContext = {
        ...context,
        productDetails: updatedProductDetails,
        buyer1Address: buyer1Acc.address,
        buyer2Address: buyer2Acc.address
    };
    fs.writeFileSync(path.join(__dirname, "demo_context.json"), JSON.stringify(newDemoContext, null, 4));
    console.log("\nDemo context updated with new owners and purchase prices.");

    console.log("--- 03: Marketplace and Purchase Scenarios Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

