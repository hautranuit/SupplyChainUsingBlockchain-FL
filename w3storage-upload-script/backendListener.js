// backendListener.js - Blockchain Event Listener with Web3.Storage Integration

import { ethers } from "ethers";
import dotenv from "dotenv";
import * as Client from "@web3-storage/w3up-client";
import { StoreMemory } from "@web3-storage/w3up-client/stores/memory";
import * as Proof from "@web3-storage/w3up-client/proof";
import { Signer } from "@web3-storage/w3up-client/principal/ed25519";
import { Blob, File } from "@web-std/file";
import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

// Load SupplyChainNFT.json manually - THIS IS THE DEPLOYED CONTRACT
const __filename_listener = fileURLToPath(import.meta.url);
const __dirname_listener = path.dirname(__filename_listener);
// Correct path to the deployed contract's artifact
const supplyChainNFTArtifactPath = path.resolve(__dirname_listener, "../SupplyChain_dapp/artifacts/contracts/SupplyChainNFT.sol/SupplyChainNFT.json");
const SupplyChainNFTArtifact = JSON.parse(fs.readFileSync(supplyChainNFTArtifactPath, 'utf-8'));

dotenv.config({ path: path.resolve(__dirname_listener, "./ifps_qr.env") });

const SupplyChainNFT_ABI = SupplyChainNFTArtifact.abi;

// --- Configuration ---
const RPC_URL = process.env.POLYGON_AMOY_RPC || process.env.AMOY_RPC_URL;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS; // This should be the SupplyChainNFT address
const BACKEND_PRIVATE_KEY = process.env.PRIVATE_KEY || process.env.BACKEND_PRIVATE_KEY;
const W3UP_KEY = process.env.KEY;
const W3UP_PROOF = process.env.PROOF;

let ipfsClient;

// --- Helper Functions (initWeb3Storage, uploadToIPFS, downloadFromIPFS - unchanged) ---
async function initWeb3Storage() {
    if (!W3UP_KEY || !W3UP_PROOF) {
        throw new Error("❌ Missing Web3.Storage KEY or PROOF in environment variables.");
    }
    const principal = Signer.parse(W3UP_KEY);
    const store = new StoreMemory();
    const client = await Client.create({ principal, store });
    const proof = await Proof.parse(W3UP_PROOF);
    try {
        const space = await client.addSpace(proof);
        await client.setCurrentSpace(space.did());
        console.log("✅ Web3.Storage client initialized and space set.");
    } catch (err) {
        if (err.message.includes("space already registered")) {
            console.warn("⚠️ Web3.Storage space already registered. Attempting to set current space.");
            const spaces = await client.spaces();
            if (spaces.length > 0) {
                await client.setCurrentSpace(spaces[0].did());
                console.log(`✅ Web3.Storage current space set to: ${spaces[0].did()}`);
            } else {
                throw new Error("❌ Web3.Storage space already registered, but no spaces found to set as current.");
            }
        } else {
            throw err;
        }
    }
    return client;
}

async function uploadToIPFS(data, filename = "product_data.json") {
    if (!ipfsClient) throw new Error("IPFS client not initialized");
    console.log(`[IPFS] Uploading data (${filename}):`, JSON.stringify(data, null, 2));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const file = new File([blob], filename);
    const cid = await ipfsClient.uploadFile(file);
    console.log(`[IPFS] Data uploaded. CID: ${cid.toString()}`);
    return cid.toString();
}

async function downloadFromIPFS(cid) {
    if (!ipfsClient) throw new Error("IPFS client not initialized");
    if (!cid || cid.trim() === "") {
        console.warn("[IPFS] Attempted to download from null or empty CID. Returning default structure.");
        return { historyLog: [] };
    }
    console.log(`[IPFS] Downloading data from CID: ${cid}`);
    const url = `https://${cid}.ipfs.w3s.link`;
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch from IPFS gateway ${url}: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log("[IPFS] Data downloaded successfully.");
        return data;
    } catch (error) {
        console.error(`[IPFS] Error downloading from CID ${cid}:`, error);
        console.warn("[IPFS] Returning default structure due to download error.");
        return { historyLog: [] };
    }
}

// --- Main Listener Logic ---
async function main() {
    if (!BACKEND_PRIVATE_KEY) {
        console.error("❌ BACKEND_PRIVATE_KEY is not set in environment variables.");
        process.exit(1);
    }
    if (!CONTRACT_ADDRESS || CONTRACT_ADDRESS === "YOUR_DEPLOYED_NFT_CORE_ADDRESS" || CONTRACT_ADDRESS === "") {
        console.error("❌ CONTRACT_ADDRESS is not set or is a placeholder.");
        process.exit(1);
    }
    if (!RPC_URL || RPC_URL === "YOUR_AMOY_RPC_URL" || RPC_URL === "") {
        console.error("❌ RPC_URL is not set or is a placeholder.");
        process.exit(1);
    }

    try {
        ipfsClient = await initWeb3Storage();
    } catch (error) {
        console.error("🚨 Failed to initialize Web3.Storage client:", error);
        process.exit(1);
    }

    const provider = new ethers.JsonRpcProvider(RPC_URL);
    const wallet = new ethers.Wallet(BACKEND_PRIVATE_KEY, provider);
    // Use the correct ABI for the deployed contract
    const supplyChainContract = new ethers.Contract(CONTRACT_ADDRESS, SupplyChainNFT_ABI, wallet);

    console.log(`👂 Listening for events on SupplyChainNFT contract at ${CONTRACT_ADDRESS} on network ${RPC_URL}...`);

    // Listener for ProductMinted (from NFTCore.sol, inherited by SupplyChainNFT)
    supplyChainContract.on("ProductMinted", async (tokenId, owner, uniqueProductID, batchNumber, manufacturingDate, event) => {
        console.log(`
--- 🎉 ProductMinted Event Received ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   Owner: ${owner}`);
        console.log(`   Unique Product ID: ${uniqueProductID}`);

        try {
            console.log(`   Fetching full RFID data for token ${tokenId}...`);
            // rfidDataMapping is in NFTCore, accessible via supplyChainContract
            const rfidData = await supplyChainContract.rfidDataMapping(tokenId.toString());
            
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
                images: [], 
                videos: [], 
                historyLog: [
                    {
                        timestamp: Math.floor(Date.now() / 1000),
                        event: "Product Minted & Registered",
                        actor: owner,
                        details: `Initial product registration. Unique ID: ${rfidData.uniqueProductID}, Batch: ${rfidData.batchNumber}.`
                    }
                ]
            };

            console.log("   Uploading initial metadata and history to IPFS...");
            const initialCid = await uploadToIPFS(initialIpfsData, `token_${tokenId}_initial_metadata.json`);

            console.log(`   Calling storeInitialCID(${tokenId}, ${initialCid}) on contract...`);
            // storeInitialCID is in NFTCore, accessible via supplyChainContract
            const tx = await supplyChainContract.storeInitialCID(tokenId.toString(), initialCid);
            console.log(`   Transaction sent: ${tx.hash}`);
            const receipt = await tx.wait();
            console.log(`   ✅ Transaction confirmed. Initial CID stored for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);

        } catch (error) {
            console.error(`   ❌ Error processing ProductMinted for token ${tokenId}:`, error);
        }
        console.log(`--- End ProductMinted Event ---`);
    });

    // Events that require history update. Ensure these events exist in SupplyChainNFT.json ABI
    const eventsToUpdateHistory = [
        // From Marketplace.sol (inherited by SupplyChainNFT)
        { eventName: "ProductListedForSale", actorField: "seller", detailsFn: (args) => `Product listed for sale by ${args.seller} at price ${ethers.formatEther(args.price)} ETH.` },
        { eventName: "PurchaseInitiated", actorField: "buyer", detailsFn: (args) => `Purchase initiated by ${args.buyer} from ${args.seller} for ${ethers.formatEther(args.price)} ETH.` },
        { eventName: "CollateralDepositedForPurchase", actorField: "buyer", detailsFn: (args) => `Collateral (payment) of ${ethers.formatEther(args.amount)} ETH deposited by ${args.buyer}. NFT transferred to buyer.` },
        { eventName: "PaymentAndTransferCompleted", actorField: "buyer", detailsFn: (args) => `Payment of ${ethers.formatEther(args.price)} ETH released to seller ${args.seller} by buyer ${args.buyer}. Purchase complete.` },
        { eventName: "ReceiptConfirmed", actorField: "confirmer", detailsFn: (args) => `Product receipt confirmed by ${args.confirmer}.` },
        // From NFTCore.sol (inherited by SupplyChainNFT)
        { eventName: "TransportStarted", actorField: "owner", detailsFn: (args) => `Transport started by ${args.owner}. From: ${args.startLocation}, To: ${args.endLocation}, Distance: ${args.distance}km, Transporters: ${args.transporters.join(", ")}.` },
        { eventName: "TransportCompleted", actorField: "completer", detailsFn: (args) => `Transport completed by ${args.completer}.` },
        // From SupplyChainNFT.sol itself
        { eventName: "DirectSaleAndTransferCompleted", actorField: "seller", detailsFn: (args) => `Direct sale and transfer from ${args.seller} to ${args.buyer} for ${ethers.formatEther(args.price)} ETH. Verified against CID: ${args.oldCIDForVerification}.` },
        // From DisputeResolution.sol (inherited by SupplyChainNFT)
        { eventName: "DisputeInitiated", actorField: "initiator", detailsFn: (args) => `Dispute initiated by ${args.initiator} regarding product owned by ${args.currentOwner}.` }
        // Add other relevant events if needed, ensuring they are in SupplyChainNFT's ABI
    ];

    eventsToUpdateHistory.forEach(({ eventName, actorField, detailsFn }) => {
        supplyChainContract.on(eventName, async (...args) => {
            const eventData = args[args.length - 1]; 
            const tokenId = args[0].toString(); 
            const actor = args[eventData.fragment.inputs.findIndex(input => input.name === actorField)];
            const details = detailsFn(eventData.args);

            console.log(`
--- 🔄 ${eventName} Event Received ---`);
            console.log(`   Token ID: ${tokenId}`);
            console.log(`   Actor (${actorField}): ${actor}`);
            console.log(`   Details: ${details}`);

            try {
                // cidMapping is in NFTCore, accessible via supplyChainContract
                const currentCid = await supplyChainContract.cidMapping(tokenId);
                if (!currentCid || currentCid.trim() === "") { // Added trim() for safety
                    console.warn(`   ⚠️ No initial CID found for token ${tokenId}. Cannot update history. This might happen if ProductMinted processing failed or is pending.`);
                    return;
                }

                const currentHistoryData = await downloadFromIPFS(currentCid);
                const newHistoryLogEntry = {
                    timestamp: Math.floor(Date.now() / 1000),
                    event: eventName,
                    actor: actor,
                    details: details,
                    transactionHash: eventData.log.transactionHash 
                };

                const updatedHistoryData = {
                    ...currentHistoryData, 
                    historyLog: [...(currentHistoryData.historyLog || []), newHistoryLogEntry]
                };

                const newCid = await uploadToIPFS(updatedHistoryData, `token_${tokenId}_history_${Date.now()}.json`);
                
                console.log(`   Calling updateProductHistoryCID(${tokenId}, ${newCid}) on contract...`);
                // updateProductHistoryCID is in NFTCore, accessible via supplyChainContract
                const tx = await supplyChainContract.updateProductHistoryCID(tokenId, newCid);
                console.log(`   Transaction sent: ${tx.hash}`);
                const receipt = await tx.wait();
                console.log(`   ✅ Transaction confirmed. History CID updated for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);

            } catch (error) {
                console.error(`   ❌ Error processing ${eventName} for token ${tokenId}:`, error);
            }
            console.log(`--- End ${eventName} Event ---`);
        });
    });

    // Listener for CIDToHistoryStored (from NFTCore.sol - for logging/confirmation)
    supplyChainContract.on("CIDToHistoryStored", (tokenId, newCid, event) => {
        console.log(`
--- ℹ️ CIDToHistoryStored Event Received (Confirmation) ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   New History CID: ${newCid}`);
        console.log(`--- End CIDToHistoryStored Event ---`);
    });

    console.log("\n⏳ Waiting for events...");
    await new Promise(() => { }); 

}

main().catch((error) => {
    console.error("🚨 Unhandled error in main listener:", error);
    process.exit(1);
});

