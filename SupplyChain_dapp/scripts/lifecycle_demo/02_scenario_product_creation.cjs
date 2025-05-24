const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");
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

// Helper function to read .env file and get a specific variable
function getEnvVariable(variableName) {
    const envFilePath = path.join(__dirname, "..", "..", "..", "w3storage-upload-script", "ifps_qr.env");
    if (!fs.existsSync(envFilePath)) {
        console.error(`Error: .env file not found at ${envFilePath}`);
        process.exit(1);
    }
    const envConfig = dotenv.parse(fs.readFileSync(envFilePath));
    if (envConfig[variableName]) {
        return envConfig[variableName];
    }
    console.error(`Error: ${variableName} not found in ${envFilePath}`);
    process.exit(1);
}
// --- End Helper Function ---

async function main() {
    console.log("--- Starting 02: Product Creation (Minting NFTs) - NO CID STORAGE ---");

    // Load context to get contract address and participant addresses
    let initialContext = {};
    if (fs.existsSync(contextFilePath)) {
        try {
            initialContext = JSON.parse(fs.readFileSync(contextFilePath, 'utf8'));
        } catch (error) {
            console.error(`Error reading initial context from ${contextFilePath}:`, error);
            process.exit(1);
        }
    }

    const contractAddress = initialContext.contractAddress;
    if (!contractAddress) {
        console.error("Error: Contract address not found in demo_context.json. Run script 01 first.");
        process.exit(1);
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    const deployer = signers.find(s => s.address.toLowerCase() === initialContext.nodes[Object.keys(initialContext.nodes).find(k => initialContext.nodes[k].name === "Deployer/Admin")].address);
    const manufacturerAcc = signers.find(s => s.address.toLowerCase() === initialContext.nodes[Object.keys(initialContext.nodes).find(k => initialContext.nodes[k].name === "Manufacturer")].address);

    if (!deployer || !manufacturerAcc) {
        console.error("Error: Could not find deployer or manufacturer account based on demo_context.json.");
        process.exit(1);
    }

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log(`Connected to contract. Manufacturer account: ${manufacturerAcc.address}`);

    const mintedProductsData = []; // To store details for context update

    const productsToMint = [
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_001",
            batchNumber: "B_ALPHA_001",
            manufacturingDate: String(Math.floor(new Date("2025-05-10").getTime() / 1000)),
            expirationDate: String(Math.floor(new Date("2027-05-10").getTime() / 1000)),
            productType: "Electronics - HighEnd Laptop",
            manufacturerID: "MANU_ACME_CORP",
            quickAccessURL: "http://example.com/products/laptop001_qa",
            nftReference: "" // Intentionally empty - CID will be added later
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_002",
            batchNumber: "B_BETA_002",
            manufacturingDate: String(Math.floor(new Date("2025-06-15").getTime() / 1000)),
            expirationDate: String(Math.floor(new Date("2026-06-15").getTime() / 1000)),
            productType: "Pharmaceuticals - Vaccine Batch",
            manufacturerID: "MANU_HEALTHCARE_INC",
            quickAccessURL: "http://example.com/products/vaccine002_qa",
            nftReference: ""
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_003",
            batchNumber: "B_GAMMA_003",
            manufacturingDate: String(Math.floor(new Date("2025-07-20").getTime() / 1000)),
            expirationDate: String(Math.floor(new Date("2025-12-20").getTime() / 1000)),
            productType: "Luxury Goods - Designer Handbag",
            manufacturerID: "MANU_FASHION_LUXE",
            quickAccessURL: "http://example.com/products/handbag003_qa",
            nftReference: ""
        }
    ];

    console.log(`\nMinting ${productsToMint.length} NFTs...`);
    for (let i = 0; i < productsToMint.length; i++) {
        const mintParams = productsToMint[i];
        console.log(`  Minting NFT ${i + 1} (${mintParams.uniqueProductID}) for ${mintParams.recipient}...`);

        const gasOptions = {
            maxPriorityFeePerGas: ethers.parseUnits('25', 'gwei'),
            maxFeePerGas: ethers.parseUnits('40', 'gwei')
        };

        let tokenId = null;
        let mintSuccessful = false;

        try {
            const txMint = await supplyChainNFT.connect(deployer).mintNFT(mintParams, gasOptions);
            const receiptMint = await txMint.wait(1);
            const block = await ethers.provider.getBlock(receiptMint.blockNumber);
            const timestamp = block.timestamp;

            let mintEvent;
            for (const log of receiptMint.logs) {
                try {
                    const parsedLog = supplyChainNFT.interface.parseLog(log);
                    if (parsedLog && parsedLog.name === "ProductMinted") {
                        mintEvent = parsedLog;
                        break;
                    }
                } catch (error) { /* Ignore */ }
            }

            if (mintEvent && mintEvent.args && mintEvent.args.tokenId !== undefined) {
                tokenId = mintEvent.args.tokenId.toString();
                console.log(`    NFT ${i + 1} Minted! Token ID: ${tokenId}, Gas Used: ${receiptMint.gasUsed.toString()}`);
                mintSuccessful = true;

                // Prepare data for context update
                mintedProductsData.push({
                    tokenId: tokenId,
                    details: mintParams,
                    currentOwnerAddress: mintParams.recipient.toLowerCase(),
                    status: "Minted",
                    history: [
                        {
                            event: "Minted",
                            actor: deployer.address.toLowerCase(),
                            recipient: mintParams.recipient.toLowerCase(),
                            timestamp: timestamp,
                            details: `Product ${mintParams.uniqueProductID} minted.`,
                            txHash: receiptMint.transactionHash
                        }
                    ],
                    lastUpdateTimestamp: timestamp
                });

            } else {
                console.error(`    ERROR: ProductMinted event not found or tokenId missing for NFT ${i + 1}. Gas Used: ${receiptMint.gasUsed.toString()}`);
                mintedProductsData.push({ tokenId: null, details: mintParams, status: "MintFailed" });
            }
        } catch (error) {
             console.error(`    ERROR minting NFT ${i + 1} (${mintParams.uniqueProductID}):`, error);
             mintedProductsData.push({ tokenId: null, details: mintParams, status: "MintFailed" });
             // Continue to next mint attempt
             continue;
        }

        // *** REMOVED storeInitialCID call ***

    }

    const successfullyMintedProducts = mintedProductsData.filter(p => p.status === "Minted");
    const failedProducts = mintedProductsData.filter(p => p.status !== "Minted");

    if (failedProducts.length > 0) {
        console.warn(`\nWarning: ${failedProducts.length} out of ${productsToMint.length} products encountered errors during minting.`);
        failedProducts.forEach(p => console.warn(`  - Product ${p.details.uniqueProductID}: Mint Status: ${p.status}`));
    }

    if (successfullyMintedProducts.length > 0) {
        console.log(`\nUpdating context for ${successfullyMintedProducts.length} successfully minted products...`);
        console.log("Successfully Minted Token IDs:", successfullyMintedProducts.map(p => p.tokenId));

        // Update demo_context.json with successfully minted product details
        readAndUpdateContext(currentContext => {
            // *** MODIFICATION START: Clear previous product data ***
            console.log("Clearing previous product data from context...");
            currentContext.products = {};
            // *** MODIFICATION END ***

            successfullyMintedProducts.forEach(product => {
                const manufacturingDateNum = parseInt(product.details.manufacturingDate, 10);
                const expirationDateNum = parseInt(product.details.expirationDate, 10);

                currentContext.products[product.tokenId] = {
                    tokenId: product.tokenId,
                    uniqueProductID: product.details.uniqueProductID,
                    batchNumber: product.details.batchNumber,
                    manufacturingDate: isNaN(manufacturingDateNum) ? null : manufacturingDateNum,
                    expirationDate: isNaN(expirationDateNum) ? null : expirationDateNum,
                    productType: product.details.productType,
                    manufacturerID: product.details.manufacturerID,
                    manufacturerAddress: product.details.recipient.toLowerCase(),
                    currentOwnerAddress: product.currentOwnerAddress,
                    status: product.status,
                    nftReference: "", // Set to empty string, will be updated by script 03
                    quickAccessURL: product.details.quickAccessURL,
                    history: product.history,
                    lastUpdateTimestamp: product.lastUpdateTimestamp,
                    price: null,
                    isListed: false,
                    transportInfo: null,
                    batchProcessingInfo: null,
                    disputeInfo: null
                };
            });

            // Update manufacturer node interactions for successful products
            if (currentContext.nodes && currentContext.nodes[manufacturerAcc.address.toLowerCase()]) {
                 if (!currentContext.nodes[manufacturerAcc.address.toLowerCase()].interactions) {
                     currentContext.nodes[manufacturerAcc.address.toLowerCase()].interactions = [];
                 }
                successfullyMintedProducts.forEach(product => {
                     currentContext.nodes[manufacturerAcc.address.toLowerCase()].interactions.push({
                        type: "MintProduct",
                        tokenId: product.tokenId,
                        timestamp: product.lastUpdateTimestamp,
                        details: `Minted product ${product.details.uniqueProductID}`,
                        txHash: product.history.find(h => h.event === "Minted")?.txHash
                     });
                });
            }
            return currentContext;
        });
    } else {
        console.warn("\nNo products were successfully minted. Context file not updated with new products.");
    }

    console.log("--- 02: Product Creation Complete (No CID Storage Attempted) ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("Script execution failed:", error);
        process.exit(1);
    });


