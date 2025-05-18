const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv"); // Added for .env loading

// Helper function to read .env file and get a specific variable
function getEnvVariable(variableName) {
    // Path to ifps_qr.env from the perspective of this script
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
            quickAccessURL: "http://example.com/products/laptop001_qa", // distinct from nftReference
            nftReference: "http://example.com/products/DEMO_PROD_001" // HTTP path as nftReference
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_002",
            batchNumber: "B_BETA_002",
            manufacturingDate: "2025-06-15",
            expirationDate: "2026-06-15",
            productType: "Pharmaceuticals - Vaccine Batch",
            manufacturerID: "MANU_HEALTHCARE_INC",
            quickAccessURL: "http://example.com/products/vaccine002_qa", // distinct from nftReference
            nftReference: "http://example.com/products/DEMO_PROD_002" // HTTP path as nftReference
        },
        {
            recipient: manufacturerAcc.address,
            uniqueProductID: "DEMO_PROD_003",
            batchNumber: "B_GAMMA_003",
            manufacturingDate: "2025-07-20",
            expirationDate: "2025-12-20",
            productType: "Luxury Goods - Designer Handbag",
            manufacturerID: "MANU_FASHION_LUXE",
            quickAccessURL: "http://example.com/products/handbag003_qa", // distinct from nftReference
            nftReference: "http://example.com/products/DEMO_PROD_003" // HTTP path as nftReference
        }
    ];

    console.log(`\nMinting ${productsToMint.length} NFTs...`);
    for (let i = 0; i < productsToMint.length; i++) {
        const mintParams = productsToMint[i];
        console.log(`  Minting NFT ${i + 1} (${mintParams.uniqueProductID}) for ${mintParams.recipient}...`);
        
        // Set explicit gas parameters to avoid underpricing
        const gasOptions = {
            maxPriorityFeePerGas: ethers.parseUnits('30', 'gwei'), // Tip for the miner, e.g., 30 Gwei
            maxFeePerGas: ethers.parseUnits('100', 'gwei')          // Max total fee, e.g., 100 Gwei
        };

        const tx = await supplyChainNFT.connect(deployer).mintNFT(mintParams, gasOptions);
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
            productsToMint[i].tokenId = tokenId.toString(); 
            console.log(`    NFT ${i + 1} Minted! Token ID: ${tokenId.toString()}, Gas Used: ${receipt.gasUsed.toString()}`);
            
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
        productDetails: productsToMint.map(p => ({ 
            recipient: p.recipient,
            uniqueProductID: p.uniqueProductID,
            batchNumber: p.batchNumber,
            manufacturingDate: p.manufacturingDate,
            expirationDate: p.expirationDate,
            productType: p.productType,
            manufacturerID: p.manufacturerID,
            quickAccessURL: p.quickAccessURL,
            nftReference: p.nftReference, // This will be the HTTP URL
            tokenId: p.tokenId, 
            currentOwnerAddress: p.recipient
            // Removed initialCID: p.nftReference
        }))
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

