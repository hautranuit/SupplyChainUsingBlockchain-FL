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

// --- Rate Limiter Class ---
class BlockchainRateLimiter {
    constructor(requestsPerSecond = 8, requestsPerMinute = 150) { // Increased limits
        this.requestsPerSecond = requestsPerSecond;
        this.requestsPerMinute = requestsPerMinute;
        this.secondWindow = [];
        this.minuteWindow = [];
    }

    async waitIfNeeded() {
        const now = Date.now();
        
        // Clean old entries
        this.secondWindow = this.secondWindow.filter(time => now - time < 1000);
        this.minuteWindow = this.minuteWindow.filter(time => now - time < 60000);
        
        // Check if we need to wait
        const needsSecondWait = this.secondWindow.length >= this.requestsPerSecond;
        const needsMinuteWait = this.minuteWindow.length >= this.requestsPerMinute;
        
        // Fast path: if no limits are hit, just record and return immediately
        if (!needsSecondWait && !needsMinuteWait) {
            const requestTime = Date.now();
            this.secondWindow.push(requestTime);
            this.minuteWindow.push(requestTime);
            return;
        }
        
        if (needsSecondWait) {
            const oldestSecond = Math.min(...this.secondWindow);
            const waitTime = 1000 - (now - oldestSecond) + 50; // Reduced buffer to 50ms
            if (waitTime > 0) {
                console.log(`   🚦 Rate limiter: Waiting ${waitTime}ms to respect per-second limit...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
        
        if (needsMinuteWait) {
            const oldestMinute = Math.min(...this.minuteWindow);
            const waitTime = 60000 - (now - oldestMinute) + 200; // Reduced buffer to 200ms
            if (waitTime > 0) {
                console.log(`   🚦 Rate limiter: Waiting ${waitTime}ms to respect per-minute limit...`);
                await new Promise(resolve => setTimeout(resolve, waitTime));
            }
        }
        
        // Record this request
        const requestTime = Date.now();
        this.secondWindow.push(requestTime);
        this.minuteWindow.push(requestTime);
    }
}

// Create rate limiter instance
const rateLimiter = new BlockchainRateLimiter();

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

async function uploadRawFileToIPFS(filePath, mimeType, retries = 3, delayMs = 1000) {
    if (!ipfsClient) throw new Error("IPFS client not initialized");
    const absoluteFilePath = path.resolve(__dirname_listener, filePath);
    console.log(`[IPFS] Attempting to upload raw file from path: ${absoluteFilePath}`);
    for (let i = 0; i < retries; i++) {
        try {
            if (!fs.existsSync(absoluteFilePath)) {
                console.error(`[IPFS] File not found at path: ${absoluteFilePath}`);
                return null;
            }
            const fileBuffer = fs.readFileSync(absoluteFilePath);
            const blob = new Blob([fileBuffer], { type: mimeType });
            const file = new File([blob], path.basename(filePath));
            const cid = await ipfsClient.uploadFile(file);
            console.log(`[IPFS] Raw file ${path.basename(filePath)} uploaded successfully. CID: ${cid.toString()}`);
            return cid.toString();
        } catch (error) {
            console.error(`[IPFS] Error uploading raw file ${filePath} (attempt ${i + 1}/${retries}). Error:`, error.message);
            if (error.code === 'ECONNRESET' && i < retries - 1) {
                console.log(`   Retrying in ${delayMs / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
                delayMs *= 2; // Exponential backoff
            } else if (i === retries - 1) {
                console.error(`[IPFS] Failed to upload raw file ${filePath} after ${retries} attempts.`);
                return null;
            } else {
                // For errors other than ECONNRESET, or if it's the last retry for ECONNRESET
                return null; // Or rethrow if you want the main loop to handle it differently
            }
        }
    }
    return null; // Should be unreachable if logic is correct
}

async function uploadToIPFS(data, filename = "product_data.json", retries = 3, delayMs = 1000) {
    if (!ipfsClient) throw new Error("IPFS client not initialized");
    console.log(`[IPFS] Uploading data to file: ${filename}`);
    for (let i = 0; i < retries; i++) {
        try {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const file = new File([blob], filename);
            const cid = await ipfsClient.uploadFile(file);
            console.log(`[IPFS] Data uploaded (${filename}). CID: ${cid.toString()}`);
            return cid.toString();
        } catch (error) {
            console.error(`[IPFS] Error uploading data for ${filename} (attempt ${i + 1}/${retries}). Error:`, error.message);
            if (error.code === 'ECONNRESET' && i < retries - 1) {
                console.log(`   Retrying in ${delayMs / 1000}s...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
                delayMs *= 2; // Exponential backoff
            } else if (i === retries - 1) {
                console.error(`[IPFS] Failed to upload data for ${filename} after ${retries} attempts.`);
                return null;
            } else {
                return null;
            }
        }
    }
    return null; // Should be unreachable
}

async function downloadFromIPFS(cid, retries = 3, delayMs = 1000) {
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

    const provider = new ethers.JsonRpcProvider(RPC_URL, {
        name: "polygon-amoy",
        chainId: 80002
    });
    
    const wallet = new ethers.Wallet(BACKEND_PRIVATE_KEY, provider);
    // Use the correct ABI for the deployed contract
    const supplyChainContract = new ethers.Contract(CONTRACT_ADDRESS, SupplyChainNFT_ABI, wallet);

    console.log(`👂 Listening for events on SupplyChainNFT contract at ${CONTRACT_ADDRESS} on network ${RPC_URL}...`);

    // --- Promise Queue for storeInitialCID operations ---
    let storeInitialCIDQueue = Promise.resolve();

    // --- Promise Queue for updateProductHistoryCID operations ---
    let updateHistoryCIDQueue = Promise.resolve(); // Initialize the new queue

    // --- Promise Queue for Dispute IPFS operations ---
    let disputeCIDUpdateQueue = Promise.resolve();

    // Listener for ProductMinted (from NFTCore.sol, inherited by SupplyChainNFT)
    supplyChainContract.on("ProductMinted", async (tokenId, owner, uniqueProductID, batchNumber, manufacturingDate, eventData) => { // Added eventData
        console.log(`\\n--- 🎉 ProductMinted Event Received ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   Owner: ${owner}`);
        console.log(`   Unique Product ID: ${uniqueProductID}`);

        // Add to the queue
        storeInitialCIDQueue = storeInitialCIDQueue.then(async () => {
            try {
                console.log(`   Processing ProductMinted for token ${tokenId} (from queue)...`);
                console.log(`   Fetching full RFID data for token ${tokenId}...`);
                
                // Apply rate limiting before blockchain read
                await rateLimiter.waitIfNeeded();
                
                // rfidDataMapping is in NFTCore, accessible via supplyChainContract
                const rfidData = await supplyChainContract.rfidDataMapping(tokenId.toString());

                let imageCid = null;
                let videoCid = null;

                // Define file paths relative to backendListener.js
                // These are assumed to be in w3storage-upload-script/images/ and w3storage-upload-script/videos/
                const imagePath = './images/product1.png'; // Corrected filename
                const videoPath = './videos/product1.mp4';

                console.log(`   Attempting to upload sample image: ${imagePath}`);
                imageCid = await uploadRawFileToIPFS(imagePath, 'image/png');
                if (imageCid) {
                    console.log(`   Sample image ${path.basename(imagePath)} uploaded. CID: ${imageCid}`);
                } else {
                    console.warn(`   ⚠️ Sample image ${path.basename(imagePath)} could not be uploaded or was not found.`);
                }

                console.log(`   Attempting to upload sample video: ${videoPath}`);
                videoCid = await uploadRawFileToIPFS(videoPath, 'video/mp4');
                if (videoCid) {
                    console.log(`   Sample video ${path.basename(videoPath)} uploaded. CID: ${videoCid}`);
                } else {
                    console.warn(`   ⚠️ Sample video ${path.basename(videoPath)} could not be uploaded or was not found.`);
                }
                
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
                    images: imageCid ? [{ name: path.basename(imagePath), cid: imageCid, uploadedAt: formatDateTime(new Date()) }] : [], // Applied formatDateTime
                    videos: videoCid ? [{ name: path.basename(videoPath), cid: videoCid, uploadedAt: formatDateTime(new Date()) }] : [], // Applied formatDateTime
                    historyLog: [
                        {
                            timestamp: formatDateTime(new Date()), // Use new formatter
                            event: "Product Minted & Registered",
                            actor: owner,
                            details: `Initial product registration. Unique ID: ${rfidData.uniqueProductID}, Batch: ${rfidData.batchNumber}.`
                        }
                    ]
                };

                console.log("   Uploading initial metadata (with media CIDs if available) and history to IPFS...");
                const initialCid = await uploadToIPFS(initialIpfsData, `token_${tokenId}_initial_metadata.json`);

                console.log(`   Calling storeInitialCID(${tokenId}, ${initialCid}) on contract...`);
                
                const gasOptions = {
                    maxPriorityFeePerGas: ethers.parseUnits('40', 'gwei'), 
                    maxFeePerGas: ethers.parseUnits('80', 'gwei')          
                };

                // Implement retry mechanism with exponential backoff
                let retries = 5;  // Maximum 5 retries
                let delay = 1000; // Start with 1 second delay (reduced from 2s)
                let tx, receipt;
                
                for(let attempt = 0; attempt < retries; attempt++) {
                    try {
                        // Apply rate limiting before blockchain request
                        await rateLimiter.waitIfNeeded();
                        
                        // storeInitialCID is in NFTCore, accessible via supplyChainContract
                        tx = await supplyChainContract.storeInitialCID(tokenId.toString(), initialCid, gasOptions);
                        console.log(`   Transaction sent for storeInitialCID (Token ${tokenId}): ${tx.hash}`);
                        
                        // Apply rate limiting before waiting for receipt
                        await rateLimiter.waitIfNeeded();
                        receipt = await tx.wait();
                        console.log(`   ✅ Transaction confirmed. Initial CID stored for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);
                        break; // Success! Break out of retry loop
                    } catch (error) {
                        if ((error.message && error.message.includes("Too Many Requests")) || 
                            (error.code === 'BAD_DATA' && error.info && error.info.payload && error.info.payload.method)) {
                            if (attempt < retries - 1) {
                                console.log(`   ⚠️ Infura rate limit hit (attempt ${attempt + 1}/${retries}). Waiting ${delay/1000}s before retrying...`);
                                await new Promise(resolve => setTimeout(resolve, delay));
                                delay *= 2; // Exponential backoff
                            } else {
                                console.error(`   ❌ Failed to store Initial CID after ${retries} attempts.`);
                                throw error; // Rethrow the error on last attempt
                            }
                        } else {
                            console.error(`   ❌ Non-rate limit error during storeInitialCID:`, error.message);
                            throw error; // Rethrow other errors immediately
                        }
                    }
                }

            } catch (error) {
                console.error(`   ❌ Error processing ProductMinted for token ${tokenId} (from queue):`, error);
            }
        }).catch(queueError => {
            // Catch errors in the queue promise itself to prevent unhandled rejections on the chain
            console.error(`   ❌ Error in storeInitialCIDQueue for token ${tokenId}:`, queueError);
            // Optionally, decide if the queue should continue or halt on such an error
            // For now, we'll let it continue with the next item.
        });
        console.log(`--- End ProductMinted Event ---`);
    });

    // Events that require history update. Ensure these events exist in SupplyChainNFT.json ABI
    const eventsToUpdateHistory = [
        // From Marketplace.sol (inherited by SupplyChainNFT)
        { eventName: "ProductListedForSale", actorField: "seller", detailsFn: (args) => `Product listed for sale by ${args.seller} at price ${ethers.formatEther(args.price)} ETH.` },
        { eventName: "PurchaseInitiated", actorField: "buyer", detailsFn: (args) => `Purchase initiated by ${args.buyer} from ${args.seller} for ${ethers.formatEther(args.price)} ETH.` },
        { eventName: "CollateralDepositedForPurchase", actorField: "buyer", detailsFn: (args) => `Collateral of ${ethers.formatEther(args.amount)} ETH deposited by ${args.buyer} for token ${args.tokenId}.` }, // ADDED THIS EVENT
        { eventName: "PaymentAndTransferCompleted", actorField: "buyer", detailsFn: (args) => `Payment of ${ethers.formatEther(args.price)} ETH released to seller ${args.seller} by buyer ${args.buyer}. Purchase complete.` },
        { eventName: "ReceiptConfirmed", actorField: "confirmer", detailsFn: (args) => `Product receipt confirmed by ${args.confirmer}.` },
        // From NFTCore.sol (inherited by SupplyChainNFT)
        { eventName: "TransportStarted", actorField: "owner", detailsFn: (args) => `Transport started by ${args.owner} for token ${args.tokenId}. Transporters: ${args.transporters.join(', ')}. From: ${args.startLocation}, To: ${args.endLocation}, Distance: ${args.distance}.` },
        { eventName: "TransportCompleted", actorField: "completer", detailsFn: (args) => `Transport leg completed by ${args.completer}.` },
        // Standard ERC721 Transfer event (from ERC721.sol, inherited via NFTCore.sol)
        { eventName: "Transfer", actorField: "to", detailsFn: (args) => `NFT transferred from ${args.from} to ${args.to}.` },
        // From SupplyChainNFT.sol itself
        { eventName: "DirectSaleAndTransferCompleted", actorField: "seller", detailsFn: (args) => `Direct sale and transfer from ${args.seller} to ${args.buyer} for ${ethers.formatEther(args.price)} ETH. Verified against CID: ${args.oldCIDForVerification}.` },
        // From DisputeResolution.sol (inherited by SupplyChainNFT)
        { eventName: "DisputeInitiated", actorField: "initiator", detailsFn: (args) => `Dispute initiated by ${args.initiator} regarding product owned by ${args.currentOwner}.` }
        // Add other relevant events if needed, ensuring they are in SupplyChainNFT\'s ABI
    ];

    // REMOVE: const transactionQueue = {}; // Key: tokenId, Value: Promise for the last transaction

    eventsToUpdateHistory.forEach(({ eventName, actorField, detailsFn }) => {
        supplyChainContract.on(eventName, async (...args) => {
            const eventData = args[args.length - 1];
            let tokenId;
            let actor;
            let details;

            // Log raw event arguments for debugging
            // console.log(`[DEBUG] Raw event args for ${eventName}:`, args);

            if (eventName === "Transfer") {
                tokenId = args[2].toString();
                const fromAddress = args[0];
                actor = args[1]; // 'to' address
                details = detailsFn({ from: fromAddress, to: args[1], tokenId: args[2] });
                if (fromAddress === ethers.ZeroAddress) {
                    console.log(`   ℹ️ Transfer event for token ${tokenId} is from ZeroAddress (minting). Skipping IPFS history update as ProductMinted handles initial CID.`);
                    console.log(`--- End ${eventName} Event ---`);
                    return;
                }
            } else if (eventName === "CollateralDepositedForPurchase") {
                tokenId = args[0].toString(); // tokenId is the first argument
                actor = args[1]; // buyer is the second argument
                details = detailsFn({ tokenId: args[0], buyer: args[1], amount: args[2] });
            }
            else {
                tokenId = args[0].toString(); // Assuming tokenId is generally the first argument
                const actorIndex = eventData.fragment.inputs.findIndex(input => input.name === actorField);
                actor = actorIndex === -1 ? 'unknown_actor' : args[actorIndex];
                details = detailsFn(eventData.args);
            }

            console.log(`\\n--- 🔄 ${eventName} Event Received ---`);
            console.log(`   Token ID: ${tokenId}`);
            console.log(`   Actor (${actorField}): ${actor}`);
            console.log(`   Details: ${details}`);
            console.log(`   Transaction Hash: ${eventData.log.transactionHash}`);

            // Add to the updateHistoryCIDQueue
            updateHistoryCIDQueue = updateHistoryCIDQueue.then(async () => {
                try {
                    console.log(`   Processing ${eventName} for token ${tokenId} (from updateHistoryCIDQueue)...`);
                    
                    // Apply rate limiting before blockchain read
                    await rateLimiter.waitIfNeeded();
                    
                    const currentCid = await supplyChainContract.cidMapping(tokenId);
                    if (!currentCid || currentCid.trim() === "") {
                        console.warn(`   ⚠️ No initial CID found for token ${tokenId} during ${eventName}. Cannot update history. This might be normal if ProductMinted hasn't completed yet or if it's an old token without prior CID.`);
                        return; // Exit this specific queued task
                    }

                    console.log(`   Current CID for token ${tokenId}: ${currentCid}`);
                    const currentHistoryData = await downloadFromIPFS(currentCid);
                    const newHistoryLogEntry = {
                        timestamp: formatDateTime(new Date()),
                        event: eventName,
                        actor: actor,
                        details: details,
                        transactionHash: eventData.log.transactionHash
                    };

                    const updatedHistoryData = {
                        ...currentHistoryData,
                        historyLog: [...(currentHistoryData.historyLog || []), newHistoryLogEntry]
                    };

                    const newCid = await uploadToIPFS(updatedHistoryData, `token_${tokenId}_history_${eventName}_${Date.now()}.json`);

                    console.log(`   Calling updateProductHistoryCID(${tokenId}, ${newCid}) on contract for event ${eventName}...`);

                    const gasOptionsUpdate = {
                        maxPriorityFeePerGas: ethers.parseUnits('60', 'gwei'), // Increased
                        maxFeePerGas: ethers.parseUnits('120', 'gwei')         // Increased
                    };

                    // Implement retry mechanism for updateProductHistoryCID
                    let retries = 5;  // Maximum 5 retries 
                    let delay = 1000; // Start with 1 second delay (reduced from 2s)
                    let tx, receipt;
                    
                    for(let attempt = 0; attempt < retries; attempt++) {
                        try {
                            // Apply rate limiting before blockchain request
                            await rateLimiter.waitIfNeeded();
                            
                            tx = await supplyChainContract.updateProductHistoryCID(tokenId, newCid, gasOptionsUpdate);
                            console.log(`   Transaction sent for ${eventName} (Token ${tokenId}): ${tx.hash}`);
                            
                            receipt = await tx.wait(1); // Wait for 1 confirmation
                            console.log(`   ✅ Transaction confirmed for ${eventName} (Token ${tokenId}). History CID updated. Gas used: ${receipt.gasUsed.toString()}`);
                            break; // Success! Break out of retry loop
                        } catch (txError) {
                            if (txError.message && txError.message.includes("Too Many Requests") && attempt < retries - 1) {
                                console.log(`   ⚠️ Infura rate limit hit (attempt ${attempt + 1}/${retries}). Waiting ${delay/1000}s before retrying...`);
                                await new Promise(resolve => setTimeout(resolve, delay));
                                delay *= 2; // Exponential backoff
                            } else {
                                console.error(`   ❌❌ Transaction Error during updateProductHistoryCID for ${eventName} (Token ${tokenId}, New CID ${newCid}):`, txError.message);
                                if (txError.code) console.error(`       Error Code: ${txError.code}`);
                                if (txError.reason) console.error(`       Reason: ${txError.reason}`);
                                if (txError.transactionHash) console.error(`       Transaction Hash (if available): ${txError.transactionHash}`);
                                
                                if (attempt === retries - 1) {
                                    console.error(`   ❌ Failed to update history CID after ${retries} attempts.`);
                                }
                                break; // Break on non-rate limit errors
                            }
                        }
                    }

                } catch (error) {
                    console.error(`   ❌ Error processing ${eventName} for token ${tokenId} (from updateHistoryCIDQueue - Outer Try-Catch):`, error.message);
                    if (error.stack) console.error(error.stack);
                }
            }).catch(queueError => {
                // Catch errors in the queue promise itself
                console.error(`   ❌ Error in updateHistoryCIDQueue for ${eventName}, token ${tokenId}:`, queueError);
            });

            console.log(`--- End ${eventName} Event ---`);
        });
    });

    // Listener for CIDToHistoryStored (from NFTCore.sol - for logging/confirmation)
    supplyChainContract.on("CIDToHistoryStored", (tokenId, newCid, eventData) => { // Added eventData
        console.log(`
--- ℹ️ CIDToHistoryStored Event Received (Confirmation) ---`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   New History CID: ${newCid}`);
        console.log(`--- End CIDToHistoryStored Event ---`);
    });

    // Listener for DisputeOpened
    // Assumes event DisputeOpened(uint256 indexed disputeId, uint256 indexed tokenId, address indexed plaintiff, string reason, string evidenceDataString, uint256 timestamp)
    supplyChainContract.on("DisputeOpened", async (disputeId, tokenId, plaintiff, reason, evidenceDataStringOrCID, timestamp, eventData) => {
        console.log(`\\n--- 🏛️ DisputeOpened Event Received ---`);
        console.log(`   Dispute ID: ${disputeId.toString()}`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   Plaintiff: ${plaintiff}`);
        console.log(`   Reason: ${reason}`);
        console.log(`   Evidence Data String (or placeholder CID): ${(evidenceDataStringOrCID || '').substring(0, 100)}...`);
        console.log(`   Transaction Hash: ${eventData.log.transactionHash}`);

        disputeCIDUpdateQueue = disputeCIDUpdateQueue.then(async () => {
            try {
                console.log(`   Processing DisputeOpened for dispute ${disputeId} (from queue)...`);
                if (!evidenceDataStringOrCID || evidenceDataStringOrCID.startsWith('ipfs://') || evidenceDataStringOrCID.length < 20) { // Basic check if it's already a CID or too short for JSON
                    console.warn(`   ⚠️ Evidence data for dispute ${disputeId} appears to be a CID already or is invalid/empty: "${evidenceDataStringOrCID}". Skipping IPFS upload by backend listener.`);
                    return;
                }

                let evidenceData;
                try {
                    evidenceData = JSON.parse(evidenceDataStringOrCID);
                } catch (parseError) {
                    console.error(`   ❌ Error parsing evidenceDataString for dispute ${disputeId}:`, parseError);
                    console.error(`      Raw evidenceDataString: ${evidenceDataStringOrCID}`);
                    return; 
                }

                const evidenceFilename = `dispute_${disputeId}_evidence.json`;
                console.log(`   Uploading evidence data for dispute ${disputeId} to IPFS as ${evidenceFilename}...`);
                const actualEvidenceCID = await uploadToIPFS(evidenceData, evidenceFilename);

                if (actualEvidenceCID) {
                    console.log(`   Evidence for dispute ${disputeId} uploaded. Actual CID: ${actualEvidenceCID}`);
                    console.log(`   Calling updateDisputeEvidenceCID(${disputeId}, "ipfs://${actualEvidenceCID}") on contract...`);
                    
                    const gasOptionsDispute = { maxPriorityFeePerGas: ethers.parseUnits('40', 'gwei'), maxFeePerGas: ethers.parseUnits('80', 'gwei') }; 
                    
                    // Implement retry mechanism for updateDisputeEvidenceCID
                    let retries = 5;  // Maximum 5 retries
                    let delay = 1000; // Start with 1 second delay (reduced from 2s)
                    let tx, receipt;
                    
                    for(let attempt = 0; attempt < retries; attempt++) {
                        try {
                            // Apply rate limiting before blockchain request
                            await rateLimiter.waitIfNeeded();
                            
                            tx = await supplyChainContract.updateDisputeEvidenceCID(disputeId.toString(), `ipfs://${actualEvidenceCID}`, gasOptionsDispute);
                            console.log(`   Transaction sent for updateDisputeEvidenceCID (Dispute ${disputeId}): ${tx.hash}`);
                            receipt = await tx.wait();
                            console.log(`   ✅ Transaction confirmed. Evidence CID stored for dispute ${disputeId}. Gas used: ${receipt.gasUsed.toString()}`);
                            break; // Success! Break out of retry loop
                        } catch (txError) {
                            if (txError.message && txError.message.includes("Too Many Requests") && attempt < retries - 1) {
                                console.log(`   ⚠️ Infura rate limit hit (attempt ${attempt + 1}/${retries}). Waiting ${delay/1000}s before retrying...`);
                                await new Promise(resolve => setTimeout(resolve, delay));
                                delay *= 2; // Exponential backoff
                            } else if (attempt === retries - 1) {
                                console.error(`   ❌ Failed to update evidence CID after ${retries} attempts.`);
                                throw txError; // Rethrow the error on last attempt
                            } else {
                                throw txError; // Rethrow other errors
                            }
                        }
                    }
                } else {
                    console.error(`   ❌ Failed to upload evidence for dispute ${disputeId} to IPFS.`);
                }
            } catch (error) {
                console.error(`   ❌ Error processing DisputeOpened for dispute ${disputeId} (from queue):`, error.message);
                if (error.stack) console.error(error.stack);
                if (error.message.includes("supplyChainContract.updateDisputeEvidenceCID is not a function")) {
                    console.warn("      💡 HINT: The smart contract needs an 'updateDisputeEvidenceCID(uint256 disputeId, string memory evidenceCID)' function callable by this backend.");
                }
            }
        }).catch(queueError => {
            console.error(`   ❌ Error in disputeCIDUpdateQueue (DisputeOpened, Dispute ${disputeId}):`, queueError);
        });
        console.log(`--- End DisputeOpened Event ---`);
    });

    // Listener for DisputeDecisionRecorded
    // Assumes event DisputeDecisionRecorded(uint256 indexed disputeId, uint256 indexed tokenId, address indexed arbitrator, string resolutionDetails, string resolutionDataString, uint8 outcome, uint256 timestamp)
    supplyChainContract.on("DisputeDecisionRecorded", async (disputeId, tokenId, arbitrator, resolutionDetailsText, resolutionDataStringOrCID, outcome, timestamp, eventData) => {
        console.log(`\\n--- ⚖️ DisputeDecisionRecorded Event Received ---`);
        console.log(`   Dispute ID: ${disputeId.toString()}`);
        console.log(`   Token ID: ${tokenId.toString()}`);
        console.log(`   Arbitrator: ${arbitrator}`);
        console.log(`   Resolution Details Text: ${resolutionDetailsText}`);
        console.log(`   Resolution Data String (or placeholder CID): ${(resolutionDataStringOrCID || '').substring(0, 100)}...`);
        console.log(`   Outcome: ${outcome.toString()}`);
        console.log(`   Transaction Hash: ${eventData.log.transactionHash}`);

        disputeCIDUpdateQueue = disputeCIDUpdateQueue.then(async () => {
            try {
                console.log(`   Processing DisputeDecisionRecorded for dispute ${disputeId} (from queue)...`);
                if (!resolutionDataStringOrCID || resolutionDataStringOrCID.startsWith('ipfs://') || resolutionDataStringOrCID.length < 20) { 
                    console.warn(`   ⚠️ Resolution data for dispute ${disputeId} appears to be a CID already or is invalid/empty: "${resolutionDataStringOrCID}". Skipping IPFS upload by backend listener.`);
                    return;
                }

                let resolutionData;
                try {
                    resolutionData = JSON.parse(resolutionDataStringOrCID);
                } catch (parseError) {
                    console.error(`   ❌ Error parsing resolutionDataString for dispute ${disputeId}:`, parseError);
                    console.error(`      Raw resolutionDataString: ${resolutionDataStringOrCID}`);
                    return; 
                }

                const resolutionFilename = `dispute_${disputeId}_resolution.json`;
                console.log(`   Uploading resolution data for dispute ${disputeId} to IPFS as ${resolutionFilename}...`);
                const actualResolutionCID = await uploadToIPFS(resolutionData, resolutionFilename);

                if (actualResolutionCID) {
                    console.log(`   Resolution data for dispute ${disputeId} uploaded. Actual CID: ${actualResolutionCID}`);
                    console.log(`   Calling updateDisputeResolutionCID(${disputeId}, "ipfs://${actualResolutionCID}") on contract...`);

                    const gasOptionsDispute = { maxPriorityFeePerGas: ethers.parseUnits('40', 'gwei'), maxFeePerGas: ethers.parseUnits('80', 'gwei') };

                    // Implement retry mechanism for updateDisputeResolutionCID
                    let retries = 5;  // Maximum 5 retries
                    let delay = 1000; // Start with 1 second delay (reduced from 2s)
                    let tx, receipt;
                    
                    for(let attempt = 0; attempt < retries; attempt++) {
                        try {
                            // Apply rate limiting before blockchain request
                            await rateLimiter.waitIfNeeded();
                            
                            tx = await supplyChainContract.updateDisputeResolutionCID(disputeId.toString(), `ipfs://${actualResolutionCID}`, gasOptionsDispute);
                            console.log(`   Transaction sent for updateDisputeResolutionCID (Dispute ${disputeId}): ${tx.hash}`);
                            receipt = await tx.wait();
                            console.log(`   ✅ Transaction confirmed. Resolution CID stored for dispute ${disputeId}. Gas used: ${receipt.gasUsed.toString()}`);
                            break; // Success! Break out of retry loop
                        } catch (txError) {
                            if (txError.message && txError.message.includes("Too Many Requests") && attempt < retries - 1) {
                                console.log(`   ⚠️ Infura rate limit hit (attempt ${attempt + 1}/${retries}). Waiting ${delay/1000}s before retrying...`);
                                await new Promise(resolve => setTimeout(resolve, delay));
                                delay *= 2; // Exponential backoff
                            } else if (attempt === retries - 1) {
                                console.error(`   ❌ Failed to update resolution CID after ${retries} attempts.`);
                                throw txError; // Rethrow the error on last attempt
                            } else {
                                throw txError; // Rethrow other errors
                            }
                        }
                    }
                } else {
                    console.error(`   ❌ Failed to upload resolution data for dispute ${disputeId} to IPFS.`);
                }
            } catch (error) {
                console.error(`   ❌ Error processing DisputeDecisionRecorded for dispute ${disputeId} (from queue):`, error.message);
                if (error.stack) console.error(error.stack);
                if (error.message.includes("supplyChainContract.updateDisputeResolutionCID is not a function")) {
                    console.warn("      💡 HINT: The smart contract needs an 'updateDisputeResolutionCID(uint256 disputeId, string memory resolutionCID)' function callable by this backend.");
                }
            }
        }).catch(queueError => {
            console.error(`   ❌ Error in disputeCIDUpdateQueue (DisputeDecisionRecorded, Dispute ${disputeId}):`, queueError);
        });
        console.log(`--- End DisputeDecisionRecorded Event ---`);
    });

    console.log("\\n⏳ Waiting for events...");
    await new Promise(() => { }); 

}

main().catch((error) => {
    console.error("🚨 Unhandled error in main listener:", error);
    process.exit(1);
});

// Helper function to format Date object to "DD/MM/YYYY, HH:MM:SS"
function formatDateTime(date) {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${day}/${month}/${year}, ${hours}:${minutes}:${seconds}`;
}

