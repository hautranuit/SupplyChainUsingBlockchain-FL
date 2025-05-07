const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function to read .env file and get a specific variable
function getEnvVariable(variableName) {
    const envFilePath = path.join(__dirname, "../../../w3storage-upload-script/ifps_qr.env");
    if (!fs.existsSync(envFilePath)) {
        console.error(`Error: .env file not found at ${envFilePath}`);
        console.error("Please run 01_deploy_and_configure.js first.");
        process.exit(1);
    }
    const envContent = fs.readFileSync(envFilePath, "utf8");
    const lines = envContent.split("\n");
    for (const line of lines) {
        if (line.startsWith(variableName + "=")) {
            return line.substring(variableName.length + 1);
        }
    }
    console.error(`Error: ${variableName} not found in ${envFilePath}`);
    process.exit(1);
}

const qrDir = path.join(__dirname, "../../../w3storage-upload-script/qr_codes");

function cleanupOldQRCodes(tokenId) {
    if (!fs.existsSync(qrDir)) {
        fs.mkdirSync(qrDir, { recursive: true });
        return;
    }
    
    const files = fs.readdirSync(qrDir);
    let cleanedCount = 0;
    
    files.forEach(file => {
        if (file.startsWith(`token_${tokenId}_`) && file.endsWith('.png')) {
            try {
                fs.unlinkSync(path.join(qrDir, file));
                cleanedCount++;
            } catch (error) {
                console.warn(`⚠️ Failed to delete old QR code ${file}: ${error.message}`);
            }
        }
    });
    
    if (cleanedCount > 0) {
        console.log(`🧹 Cleaned up ${cleanedCount} old QR code(s) for token ${tokenId}`);
    }
}

async function main() {
    console.log("--- Starting 02: Product Creation (Minting NFTs) ---");

    const contractAddress = getEnvVariable("CONTRACT_ADDRESS");
    if (!contractAddress) {
        // getEnvVariable will exit if not found, but double check
        return;
    }
    console.log(`Using SupplyChainNFT contract at: ${contractAddress}`);

    const signers = await ethers.getSigners();
    const deployer = signers[0]; // Has MINTER_ROLE from script 01
    const manufacturerAcc = signers[1]; // Configured as Manufacturer in script 01

    if (signers.length < 2) {
        console.error("This script requires at least 2 signers (deployer/minter, manufacturer).");
        process.exit(1);
    }

    const supplyChainNFT = await ethers.getContractAt("SupplyChainNFT", contractAddress, deployer);
    console.log(`Connected to contract. Manufacturer account: ${manufacturerAcc.address}`);

    const mintedTokenIds = [];
    const productsToMint = [
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_001",
            batchNumber: "B_ALPHA_001",
            manufacturingDate: "2025-05-10",
            expirationDate: "2027-05-10",
            productType: "Electronics - HighEnd Laptop",
            manufacturerID: "MANU_ACME_CORP",
            quickAccessURL: "http://example.com/products/laptop001",
            nftReference: "ipfs://QmExampleLaptopHash"
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_002",
            batchNumber: "B_BETA_002",
            manufacturingDate: "2025-06-15",
            expirationDate: "2026-06-15",
            productType: "Pharmaceuticals - Vaccine Batch",
            manufacturerID: "MANU_HEALTHCARE_INC",
            quickAccessURL: "http://example.com/products/vaccine002",
            nftReference: "ipfs://QmExampleVaccineHash"
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_003",
            batchNumber: "B_GAMMA_003",
            manufacturingDate: "2025-07-20",
            expirationDate: "2025-12-20",
            productType: "Luxury Goods - Designer Handbag",
            manufacturerID: "MANU_FASHION_LUXE",
            quickAccessURL: "http://example.com/products/handbag003",
            nftReference: "ipfs://QmExampleHandbagHash"
        }
    ];

    console.log(`\nMinting ${productsToMint.length} NFTs...`);
    for (let i = 0; i < productsToMint.length; i++) {
        const mintParams = productsToMint[i];
        console.log(`  Minting NFT ${i + 1} (${mintParams.uniqueProductID}) for ${mintParams.recipient}...`);
        const tx = await supplyChainNFT.connect(deployer).mintNFT(mintParams);
        const receipt = await tx.wait(1);
        
        let mintEvent;
        for (const log of receipt.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ProductMinted") {
                    mintEvent = parsedLog;
                    break;
                }
            } catch (error) { /* Ignore - could be other events from dependencies */ }
        }

        if (mintEvent && mintEvent.args && mintEvent.args.tokenId !== undefined) {
            const tokenId = mintEvent.args.tokenId;
            mintedTokenIds.push(tokenId.toString());
            console.log(`    NFT ${i + 1} Minted! Token ID: ${tokenId.toString()}, Gas Used: ${receipt.gasUsed.toString()}`);
            
            // Store initial CID (as done in some tests, assuming deployer has UPDATER_ROLE)
            const initialPlaceholderCID = "ipfs://placeholder_cid_for_" + mintParams.uniqueProductID;
            console.log(`    Storing initial placeholder CID for Token ID ${tokenId.toString()}: ${initialPlaceholderCID}`);
            const storeCidTx = await supplyChainNFT.connect(deployer).storeInitialCID(tokenId, initialPlaceholderCID);
            const storeCidReceipt = await storeCidTx.wait(1);
            console.log(`      Initial CID stored. Gas Used: ${storeCidReceipt.gasUsed.toString()}`);

            // Validate initial CID format
            if (!initialPlaceholderCID.startsWith('ipfs://')) {
                console.error(`Invalid initial CID format: ${initialPlaceholderCID}`);
                process.exit(1);
            }

        } else {
            console.error(`    ERROR: ProductMinted event not found or tokenId missing for NFT ${i + 1}. Gas Used: ${receipt.gasUsed.toString()}`);
            // Decide if we should exit or continue
        }
    }

    if (mintedTokenIds.length === productsToMint.length) {
        console.log("\nAll NFTs minted successfully!");
    } else {
        console.warn("\nWarning: Not all NFTs may have been minted successfully or token IDs captured.");
    }
    console.log("Minted Token IDs:", mintedTokenIds);

    // Save context for subsequent scripts
    const demoContextPath = path.join(__dirname, "demo_context.json");
    const demoContext = {
        contractAddress: contractAddress,
        manufacturerAddress: manufacturerAcc.address,
        deployerAddress: deployer.address,
        tokenIds: mintedTokenIds,
        productDetails: productsToMint.map((p, i) => ({ ...p, tokenId: mintedTokenIds[i] || null }))
    };
    fs.writeFileSync(demoContextPath, JSON.stringify(demoContext, null, 4));
    console.log(`\nDemo context (contract address, token IDs) saved to: ${demoContextPath}`);
    console.log("--- 02: Product Creation Complete ---");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

