// backendListener.js - Skeleton for Blockchain Event Listeners

import { ethers } from "ethers";
import NFTCoreArtifact from "../SupplyChain_dapp/artifacts/contracts/NFTCore.sol/NFTCore.json" assert { type: "json" };
import dotenv from "dotenv";
dotenv.config();
const NFTCoreABI = NFTCoreArtifact.abi;
// --- Configuration (Replace with your actual values) ---
const RPC_URL = process.env.POLYGON_AMOY_RPC || "YOUR_AMOY_RPC_URL"; // Or your network's RPC URL
const NFT_CORE_ADDRESS = process.env.CONTRACT_ADDRESS || "YOUR_DEPLOYED_NFT_CORE_ADDRESS";
const BACKEND_PRIVATE_KEY = process.env.PRIVATE_KEY; // Private key for the account with UPDATER_ROLE and Owner permissions
const IPFS_CLIENT = null; // Initialize your IPFS client here (e.g., using ipfs-http-client or web3.storage client)

// --- Helper Functions (Placeholders - Implement with your IPFS logic) ---
async function uploadToIPFS(data) {
    console.log("[IPFS Placeholder] Uploading data:", data);
    // Replace with your actual IPFS upload logic (e.g., using IPFS_CLIENT.add())
    // Example using web3.storage:
    // const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    // const file = new File([blob], 'product_data.json');
    // const cid = await IPFS_CLIENT.put([file]);
    const placeholderCID = "bafkrei" + Math.random().toString(36).substring(2);
    console.log("[IPFS Placeholder] Data uploaded. CID:", placeholderCID);
    return placeholderCID;
}

async function downloadFromIPFS(cid) {
    console.log("[IPFS Placeholder] Downloading data from CID:", cid);
    // Replace with your actual IPFS download logic
    // Example: Fetch from gateway or use IPFS_CLIENT.get()
    const placeholderData = {
        previousHistoryCID: null, // Link to previous history if applicable
        log: [
            { timestamp: Date.now() - 10000, event: "Manufactured", details: "Initial data" }
        ]
    };
    console.log("[IPFS Placeholder] Data downloaded:", placeholderData);
    return placeholderData;
}

// --- Main Listener Logic ---
async function main() {
    if (!BACKEND_PRIVATE_KEY) {
        console.error("❌ BACKEND_PRIVATE_KEY is not set in environment variables.");
        process.exit(1);
    }
    if (!NFT_CORE_ADDRESS || NFT_CORE_ADDRESS === "YOUR_DEPLOYED_NFT_CORE_ADDRESS") {
        console.error("❌ NFT_CORE_ADDRESS is not set or is a placeholder.");
        process.exit(1);
    }
    if (!IPFS_CLIENT) {
        console.warn("⚠️ IPFS_CLIENT is not initialized. IPFS operations will be placeholders.");
    }

    const provider = new ethers.JsonRpcProvider(RPC_URL);
    const wallet = new ethers.Wallet(BACKEND_PRIVATE_KEY, provider);
    const nftCoreContract = new ethers.Contract(NFT_CORE_ADDRESS, NFTCoreABI, wallet);

    console.log(`👂 Listening for events on NFTCore contract at ${NFT_CORE_ADDRESS}...`);

    // --- Listener for ProductMinted ---    
nftCoreContract.on("ProductMinted", async (owner, tokenId, event) => {
        console.log(`
--- 🎉 ProductMinted Event Received ---`);
        console.log(`   Owner: ${owner}`);
        console.log(`   Token ID: ${tokenId.toString()}`);

        try {
            // 1. Get RFID Data from contract
            console.log(`   Fetching RFID data for token ${tokenId}...`);
            const rfidData = await nftCoreContract.rfidDataMapping(tokenId);
            console.log("   RFID Data:", {
                uniqueProductID: rfidData.uniqueProductID,
                batchNumber: rfidData.batchNumber,
                manufacturingDate: rfidData.manufacturingDate,
                // ... add other fields as needed
            });

            // 2. Prepare initial data for IPFS
            const initialIpfsData = {
                rfid: {
                    uniqueProductID: rfidData.uniqueProductID,
                    batchNumber: rfidData.batchNumber,
                    manufacturingDate: rfidData.manufacturingDate,
                    expirationDate: rfidData.expirationDate,
                    productType: rfidData.productType,
                    manufacturerID: rfidData.manufacturerID,
                    quickAccessURL: rfidData.quickAccessURL,
                    nftReference: rfidData.nftReference
                },
                // Add links to images/videos if applicable
                images: [],
                videos: [],
                historyLog: [
                    {
                        timestamp: Math.floor(Date.now() / 1000),
                        event: "Product Minted",
                        actor: owner,
                        details: "Initial product registration and NFT creation."
                    }
                ]
            };

            // 3. Upload to IPFS
            console.log("   Uploading initial data to IPFS...");
            const initialCid = await uploadToIPFS(initialIpfsData);

            // 4. Call storeCID on the contract
            console.log(`   Calling storeCID(${tokenId}, ${initialCid}) on contract...`);
            const tx = await nftCoreContract.storeCID(tokenId, initialCid);
            console.log(`   Transaction sent: ${tx.hash}`);
            const receipt = await tx.wait();
            console.log(`   ✅ Transaction confirmed. CID stored for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);

        } catch (error) {
            console.error(`   ❌ Error processing ProductMinted for token ${tokenId}:`, error);
        }
        console.log(`--- End ProductMinted Event ---`);
    });

    // --- Listener for ReceiptConfirmed ---    
nftCoreContract.on("ReceiptConfirmed", async (tokenId, confirmer, event) => {
        console.log(`
--- ✅ ReceiptConfirmed Event Received ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   Confirmed By: ${confirmer}`);

        try {
            // 1. Get the current history CID from the contract
            console.log(`   Fetching current history CID for token ${tokenId}...`);
            const currentCid = await nftCoreContract.cidMapping(tokenId);
            if (!currentCid) {
                console.warn(`   ⚠️ No CID found for token ${tokenId}. Cannot update history.`);
                return;
            }
            console.log(`   Current CID: ${currentCid}`);

            // 2. Download current history data from IPFS
            const currentHistoryData = await downloadFromIPFS(currentCid);

            // 3. Append confirmation event to history
            const updatedHistoryData = {
                ...currentHistoryData, // Keep existing data (RFID, images, etc.)
                historyLog: [
                    ...(currentHistoryData.historyLog || []), // Keep previous logs
                    {
                        timestamp: Math.floor(Date.now() / 1000),
                        event: "Receipt Confirmed",
                        actor: confirmer,
                        details: "Customer confirmed receipt of the product."
                    }
                ]
            };

            // 4. Upload updated history to IPFS
            console.log("   Uploading updated history to IPFS...");
            const newCid = await uploadToIPFS(updatedHistoryData);

            // 5. Call updateProductHistory on the contract (requires UPDATER_ROLE)
            console.log(`   Calling updateProductHistory(${tokenId}, ${newCid}) on contract...`);
            // Ensure the wallet used has the UPDATER_ROLE granted
            const tx = await nftCoreContract.updateProductHistory(tokenId, newCid);
            console.log(`   Transaction sent: ${tx.hash}`);
            const receipt = await tx.wait();
            console.log(`   ✅ Transaction confirmed. History updated for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);

        } catch (error) {
            console.error(`   ❌ Error processing ReceiptConfirmed for token ${tokenId}:`, error);
        }
        console.log(`--- End ReceiptConfirmed Event ---`);
    });

    // --- Listener for CIDToHistoryStored (Optional: Just for logging/confirmation) ---
    nftCoreContract.on("CIDToHistoryStored", (tokenId, cid, event) => {
        console.log(`
--- ℹ️ CIDToHistoryStored Event Received ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   New History CID: ${cid}`);
        console.log(`--- End CIDToHistoryStored Event ---`);
    });

    // Keep the script running (in a real backend, this would be part of a server process)
    // For a simple script, you might just let it run indefinitely or add process handling
    console.log("\nWaiting for events...");
    // This prevents the script from exiting immediately
    await new Promise(() => { }); 

}

main().catch((error) => {
    console.error("🚨 Unhandled error in main listener:", error);
    process.exit(1);
});

