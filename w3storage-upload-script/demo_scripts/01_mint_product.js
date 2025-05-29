// 01_mint_product.js (Integrated with Media Upload)
import { ethers } from "ethers";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import * as Client from "@web3-storage/w3up-client";
import { StoreMemory } from "@web3-storage/w3up-client/stores/memory";
import * as Proof from "@web3-storage/w3up-client/proof";
import { Signer } from "@web3-storage/w3up-client/principal/ed25519";
import { File } from "@web-std/file";

// Determine the correct path to .env file and artifact based on script execution location
const __filename_script = fileURLToPath(import.meta.url);
const __dirname_script = path.dirname(__filename_script);
const envPath = path.resolve(__dirname_script, "../ifps_qr.env");
dotenv.config({ path: envPath });

// Load SupplyChainNFT.json manually
const supplyChainNFTArtifactPath = path.resolve(__dirname_script, "../../SupplyChain_dapp/artifacts/contracts/SupplyChainNFT.sol/SupplyChainNFT.json");
const SupplyChainNFTArtifact = JSON.parse(fs.readFileSync(supplyChainNFTArtifactPath, "utf-8"));
const SupplyChainNFT_ABI = SupplyChainNFTArtifact.abi;

// --- Environment Variables & Constants ---
const RPC_URL = process.env.POLYGON_AMOY_RPC || process.env.AMOY_RPC_URL;
const PRIVATE_KEY = process.env.PRIVATE_KEY || process.env.BACKEND_PRIVATE_KEY;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;
const W3UP_KEY = process.env.KEY;
const W3UP_PROOF = process.env.PROOF;

let ipfsClient;

// --- IPFS Helper Functions ---
async function initWeb3Storage() {
    if (!W3UP_KEY || !W3UP_PROOF) {
        console.error("❌ Missing Web3.Storage KEY or PROOF in .env file.");
        return null; // Allow script to proceed if IPFS is not strictly needed for mint-only
    }
    try {
        const principal = Signer.parse(W3UP_KEY);
        const store = new StoreMemory();
        const client = await Client.create({ principal, store });
        const proof = await Proof.parse(W3UP_PROOF);
        const space = await client.addSpace(proof);
        await client.setCurrentSpace(space.did());
        console.log("✅ Web3.Storage client initialized and space set for media upload.");
        return client;
    } catch (err) {
        if (err.message.includes("space already registered")) {
            try {
                const principal = Signer.parse(W3UP_KEY);
                const store = new StoreMemory();
                const client = await Client.create({ principal, store });
                const spaces = await client.spaces();
                if (spaces.length > 0) {
                    await client.setCurrentSpace(spaces[0].did());
                    console.log(`✅ Web3.Storage current space set to: ${spaces[0].did()} for media upload.`);
                    return client;
                }
            } catch (recoveryErr) {
                console.warn("⚠️ Failed to recover Web3.Storage space for media upload:", recoveryErr.message);
                return null;
            }
        }
        console.warn("⚠️ Error initializing Web3.Storage for media upload:", err.message);
        return null;
    }
}

async function uploadFileToIPFS(filePath, client) {
    if (!client) {
        console.error("[IPFS] Client not initialized. Cannot upload file.");
        throw new Error("IPFS client not initialized");
    }
    console.log(`[IPFS] Uploading file: ${filePath}`);
    try {
        const fileContent = fs.readFileSync(filePath);
        const fileName = path.basename(filePath);
        const file = new File([fileContent], fileName);
        const cid = await client.uploadFile(file);
        console.log(`[IPFS] File ${fileName} uploaded. CID: ${cid.toString()}`);
        return cid.toString();
    } catch (error) {
        console.error(`[IPFS] Error uploading file ${filePath}:`, error.message);
        throw error;
    }
}

async function uploadJsonToIPFS(jsonObject, fileName, client) {
    if (!client) {
        console.error("[IPFS] Client not initialized. Cannot upload JSON.");
        throw new Error("IPFS client not initialized");
    }
    console.log(`[IPFS] Uploading JSON data (${fileName})`);
    try {
        const jsonString = JSON.stringify(jsonObject, null, 2);
        const file = new File([jsonString], fileName, { type: "application/json" });
        const cid = await client.uploadFile(file);
        console.log(`[IPFS] JSON ${fileName} uploaded. CID: ${cid.toString()}`);
        return cid.toString();
    } catch (error) {
        console.error(`[IPFS] Error uploading JSON ${fileName}:`, error.message);
        throw error;
    }
}

async function downloadJsonFromIPFSGateway(cid) {
    if (!cid || cid.trim() === "") {
        console.error("❌ Attempted to download from null or empty CID.");
        return null;
    }
    console.log(`[IPFS Gateway] Downloading JSON from CID: ${cid}`);
    const url = `https://${cid}.ipfs.w3s.link`;
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch from IPFS gateway ${url}: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log("[IPFS Gateway] JSON data downloaded successfully.");
        return data;
    } catch (error) {
        console.error(`[IPFS Gateway] Error downloading JSON from CID ${cid}:`, error.message);
        return null;
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length < 8) {
        console.error("❌ Usage: node 01_mint_product.js <uniqueProductID> <batchNumber> <mfgDate> <expDate> <productType> <manufacturerID> <qrAccessURL> <nftRef> [imagePath] [videoPath]");
        console.error("Example (no media): node 01_mint_product.js P123 B001 2023-01-01 2025-01-01 \"Coffee Beans\" Manu001 http://example.com/qr/P123 \"InitialNFTData\"");
        console.error("Example (with media): node 01_mint_product.js P124 B002 2023-02-01 2025-02-01 \"Tea Leaves\" Manu002 http://example.com/qr/P124 \"InitialTeaData\" ../images/product1.png ../videos/product1.mp4");
        process.exit(1);
    }

    const [uniqueProductID, batchNumber, manufacturingDate, expirationDate, productType, manufacturerID, quickAccessURL, nftReference] = args.slice(0, 8);
    const imagePathArg = args.length > 8 ? args[8] : null;
    const videoPathArg = args.length > 9 ? args[9] : null;

    const imagePath = imagePathArg ? path.resolve(__dirname_script, imagePathArg) : null;
    const videoPath = videoPathArg ? path.resolve(__dirname_script, videoPathArg) : null;

    if (!RPC_URL || !PRIVATE_KEY || !CONTRACT_ADDRESS) {
        console.error("❌ Missing RPC_URL, PRIVATE_KEY, or CONTRACT_ADDRESS in .env file.");
        process.exit(1);
    }

    const provider = new ethers.JsonRpcProvider(RPC_URL);
    const wallet = new ethers.Wallet(PRIVATE_KEY, provider);
    const supplyChainContract = new ethers.Contract(CONTRACT_ADDRESS, SupplyChainNFT_ABI, wallet);

    console.log(`Attempting to mint NFT on contract ${CONTRACT_ADDRESS}:`);
    console.log(`  Unique Product ID: ${uniqueProductID}, Batch: ${batchNumber}`);
    console.log(`  Recipient (Minter): ${wallet.address}`);
    if (imagePath) console.log(`  Image Path: ${imagePath}`);
    if (videoPath) console.log(`  Video Path: ${videoPath}`);

    try {
        const mintParams = {
            recipient: wallet.address, 
            uniqueProductID,
            batchNumber,
            manufacturingDate,
            expirationDate,
            productType,
            manufacturerID,
            quickAccessURL,
            nftReference
        };

        console.log("\n⏳ Sending mintNFT transaction...");
        const tx = await supplyChainContract.mintNFT(mintParams);
        console.log(`   Transaction sent: ${tx.hash}`);
        const receipt = await tx.wait();

        if (!receipt || !receipt.logs) {
            console.error("❌ Transaction receipt not found or logs are missing.");
            process.exit(1);
        }

        let tokenId = null;
        for (const log of receipt.logs) {
            try {
                const parsedLog = supplyChainContract.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ProductMinted") {
                    tokenId = parsedLog.args.tokenId.toString();
                    break;
                }
            } catch (e) { /* Ignore */ }
        }

        if (!tokenId) {
            console.error("❌ Could not find ProductMinted event. Minting might have failed.");
            process.exit(1);
        }
        
        console.log(`✅ NFT Minted Successfully! Token ID: ${tokenId}, Gas used: ${receipt.gasUsed.toString()}`);
        console.log("   Waiting for backendListener to store initial CID and emit InitialCIDStored event...");

        // Wait for InitialCIDStored event from the contract (emitted by backendListener after processing ProductMinted)
        const initialCIDStoredPromise = new Promise((resolve, reject) => {
            const eventName = "InitialCIDStored";
            const listener = (eventTokenId, eventInitialCID, eventActor) => {
                if (eventTokenId.toString() === tokenId) {
                    console.log(`   🎉 Event ${eventName} received for Token ID ${tokenId}:`);
                    console.log(`      Initial CID: ${eventInitialCID}`);
                    console.log(`      Actor (backend): ${eventActor}`);
                    supplyChainContract.off(eventName, listener); // Remove listener
                    resolve(eventInitialCID);
                }
            };
            supplyChainContract.on(eventName, listener);
            // Timeout if event not received
            setTimeout(() => {
                supplyChainContract.off(eventName, listener);
                reject(new Error(`Timeout waiting for ${eventName} event for Token ID ${tokenId}. Ensure backendListener.js is running and configured.`));
            }, 120000); // 2 minutes timeout
        });

        const initialCID = await initialCIDStoredPromise;

        if ((imagePath && !fs.existsSync(imagePath)) || (videoPath && !fs.existsSync(videoPath))) {
            console.log("✅ Product minted with initial metadata. Media upload skipped due to missing files.");
            process.exit(0);
        }

        if (imagePath || videoPath) {
            console.log("\n--- Proceeding with Media Upload and Metadata Update ---");
            ipfsClient = await initWeb3Storage();
            if (!ipfsClient) {
                console.warn("⚠️ IPFS client failed to initialize. Cannot upload media. Product will only have initial metadata.");
                process.exit(0); // Or handle as an error depending on requirements
            }

            let uploadedImageCID = null;
            let uploadedVideoCID = null;

            if (imagePath) {
                uploadedImageCID = await uploadFileToIPFS(imagePath, ipfsClient);
            }
            if (videoPath) {
                uploadedVideoCID = await uploadFileToIPFS(videoPath, ipfsClient);
            }

            console.log("\nFetching initial metadata from IPFS gateway...");
            let productMetadata = await downloadJsonFromIPFSGateway(initialCID);
            if (!productMetadata) {
                console.error("❌ Failed to download initial product metadata. Cannot update with media. Exiting.");
                process.exit(1);
            }

            console.log("Updating metadata with media CIDs...");
            if (!productMetadata.images) productMetadata.images = [];
            if (!productMetadata.videos) productMetadata.videos = [];

            if (uploadedImageCID) {
                productMetadata.images.push({ cid: uploadedImageCID, name: path.basename(imagePath), type: "image/png" }); // Adjust type if needed
            }
            if (uploadedVideoCID) {
                productMetadata.videos.push({ cid: uploadedVideoCID, name: path.basename(videoPath), type: "video/mp4" }); // Adjust type if needed
            }
            
            if (!productMetadata.historyLog) productMetadata.historyLog = [];
            const mediaUpdateLogEntry = {
                timestamp: Math.floor(Date.now() / 1000),
                event: "Media Added During Minting",
                actor: wallet.address,
                details: `Added media during initial minting.`
            };
            if(uploadedImageCID) mediaUpdateLogEntry.imageCID = uploadedImageCID;
            if(uploadedVideoCID) mediaUpdateLogEntry.videoCID = uploadedVideoCID;
            productMetadata.historyLog.push(mediaUpdateLogEntry);
            
            console.log("Uploading updated metadata to IPFS...");
            const updatedMetadataFileName = `token_${tokenId}_comp_metadata_${Date.now()}.json`;
            const newComprehensiveCID = await uploadJsonToIPFS(productMetadata, updatedMetadataFileName, ipfsClient);

            console.log("Updating smart contract with new comprehensive metadata CID...");
            const updateTx = await supplyChainContract.updateProductHistoryCID(tokenId, newComprehensiveCID);
            console.log(`   Update transaction sent: ${updateTx.hash}`);
            const updateReceipt = await updateTx.wait();
            console.log(`   ✅ Update transaction confirmed. Gas used: ${updateReceipt.gasUsed.toString()}`);

            console.log("\n🎉 NFT Minted and Media Added Successfully!");
            console.log(`   Token ID: ${tokenId}`);
            if (uploadedImageCID) console.log(`   Uploaded Image CID: ${uploadedImageCID} (Link: https://${uploadedImageCID}.ipfs.w3s.link)`);
            if (uploadedVideoCID) console.log(`   Uploaded Video CID: ${uploadedVideoCID} (Link: https://${uploadedVideoCID}.ipfs.w3s.link)`);
            console.log(`   New Comprehensive Metadata CID: ${newComprehensiveCID} (Link: https://${newComprehensiveCID}.ipfs.w3s.link)`);
        } else {
            console.log("\n✅ NFT Minted successfully with initial metadata (no media provided).");
            console.log(`   Token ID: ${tokenId}`);
            console.log(`   Initial Metadata CID: ${initialCID} (Link: https://${initialCID}.ipfs.w3s.link)`);
        }

    } catch (error) {
        console.error("❌ Error in minting process:", error.message);
        if (error.stack) console.error(error.stack);
        process.exit(1);
    }
}

main();

