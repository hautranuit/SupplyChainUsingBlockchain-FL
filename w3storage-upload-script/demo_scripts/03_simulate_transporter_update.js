// 03_simulate_transporter_update.js
import { ethers } from "ethers";
import dotenv from "dotenv";
import path from "path";
import crypto from "crypto";
import * as Client from "@web3-storage/w3up-client";
import { StoreMemory } from "@web3-storage/w3up-client/stores/memory";
import * as Proof from "@web3-storage/w3up-client/proof";
import { Signer } from "@web3-storage/w3up-client/principal/ed25519";
import { Blob, File } from "@web-std/file";
import fs from "fs";
import { fileURLToPath } from "url";

// Determine the correct path to .env file and artifact based on script execution location
const __filename_script = fileURLToPath(import.meta.url);
const __dirname_script = path.dirname(__filename_script);
const envPath = path.resolve(__dirname_script, "../ifps_qr.env");
dotenv.config({ path: envPath });

// Load SupplyChainNFT.json manually - THIS IS THE DEPLOYED CONTRACT
const supplyChainNFTArtifactPath = path.resolve(__dirname_script, "../../SupplyChain_dapp/artifacts/contracts/SupplyChainNFT.sol/SupplyChainNFT.json");
const SupplyChainNFTArtifact = JSON.parse(fs.readFileSync(supplyChainNFTArtifactPath, "utf-8"));
const SupplyChainNFT_ABI = SupplyChainNFTArtifact.abi;

let ipfsClient; // Will be initialized in main

// --- Decryption Helper Functions (from decryptCID.js) ---
function decrypt(encryptedText, secretKey) {
    const [ivHex, encryptedHex] = encryptedText.split(":");
    const iv = Buffer.from(ivHex, "hex");
    const encryptedBuffer = Buffer.from(encryptedHex, "hex");
    const decipher = crypto.createDecipheriv("aes-256-cbc", Buffer.from(secretKey, "hex"), iv);
    let decrypted = decipher.update(encryptedBuffer, null, "utf8");
    decrypted += decipher.final("utf8");
    return decrypted;
}

function verifyHMAC(encryptedText, hmac, hmacKey) {
    const expectedHMAC = crypto.createHmac("sha256", Buffer.from(hmacKey, "hex")).update(encryptedText).digest("hex");
    return expectedHMAC === hmac;
}

// --- IPFS Helper Functions (from recordTransportLog_modified.js / backendListener.js) ---
async function initWeb3Storage() {
    const W3UP_KEY = process.env.KEY;
    const W3UP_PROOF = process.env.PROOF;
    if (!W3UP_KEY || !W3UP_PROOF) {
        throw new Error("❌ Missing Web3.Storage KEY or PROOF in environment variables for IPFS client.");
    }
    const principal = Signer.parse(W3UP_KEY);
    const store = new StoreMemory();
    const client = await Client.create({ principal, store });
    const proof = await Proof.parse(W3UP_PROOF);
    try {
        const space = await client.addSpace(proof);
        await client.setCurrentSpace(space.did());
    } catch (err) {
        if (err.message.includes("space already registered")) {
            const spaces = await client.spaces();
            if (spaces.length > 0) await client.setCurrentSpace(spaces[0].did());
            else throw new Error("❌ Web3.Storage space already registered, but no spaces found.");
        } else throw err;
    }
    console.log("✅ Web3.Storage client initialized for IPFS operations.");
    return client;
}

async function downloadFromIPFS(cid) {
    if (!ipfsClient) throw new Error("IPFS client not initialized for download");
    if (!cid || cid.trim() === "") {
        console.warn("[IPFS] Attempted to download from null/empty CID. Returning default structure.");
        return { historyLog: [] };
    }
    console.log(`[IPFS] Downloading data from CID: ${cid}`);
    const url = `https://${cid}.ipfs.w3s.link`;
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch from IPFS gateway ${url}: ${response.status} ${response.statusText}`);
        const data = await response.json();
        console.log("[IPFS] Data downloaded successfully.");
        return data;
    } catch (error) {
        console.error(`[IPFS] Error downloading from CID ${cid}:`, error);
        return { historyLog: [] };
    }
}

async function uploadToIPFS(data, filename = "product_history.json") {
    if (!ipfsClient) throw new Error("IPFS client not initialized for upload");
    console.log(`[IPFS] Uploading data (${filename}):`, JSON.stringify(data, null, 2));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const file = new File([blob], filename);
    const cid = await ipfsClient.uploadFile(file);
    console.log(`[IPFS] Data uploaded. New CID: ${cid.toString()}`);
    return cid.toString();
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length < 4) {
        console.error("❌ Usage: node 03_simulate_transporter_update.js <tokenId> \"<encryptedQrPayload>\" \"<newLocation>\" <transporterWalletAddress>");
        console.error("Example: node 03_simulate_transporter_update.js 1 \"iv:encrypted:hmac\" \"Warehouse B\" 0xTransporterAddress");
        process.exit(1);
    }
    const [tokenId, encryptedQrPayload, newLocation, transporterWalletAddress] = args;

    if (!ethers.isAddress(transporterWalletAddress)) {
        throw new Error("Invalid transporter wallet address format");
    }

    const RPC_URL = process.env.POLYGON_AMOY_RPC || process.env.AMOY_RPC_URL;
    const PRIVATE_KEY = process.env.PRIVATE_KEY || process.env.BACKEND_PRIVATE_KEY; // This key needs UPDATER_ROLE
    const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS; // This should be the deployed SupplyChainNFT address
    const AES_SECRET_KEY = process.env.AES_SECRET_KEY;
    const HMAC_SECRET_KEY = process.env.HMAC_SECRET_KEY;

    if (!RPC_URL || !PRIVATE_KEY || !CONTRACT_ADDRESS || !AES_SECRET_KEY || !HMAC_SECRET_KEY) {
        console.error("❌ Missing required environment variables (RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS, AES_SECRET_KEY, HMAC_SECRET_KEY).");
        process.exit(1);
    }

    try {
        ipfsClient = await initWeb3Storage(); // Initialize IPFS client

        // 1. Decrypt QR Payload to get current IPFS History CID
        console.log("\nDecrypting QR payload...");
        const parts = encryptedQrPayload.split(":");
        if (parts.length !== 3) throw new Error("Invalid encrypted QR payload format. Expected iv:encrypted:hmac");
        const [ivHex, encryptedHex, hmacFromPayload] = parts;
        const encryptedTextForHMAC = `${ivHex}:${encryptedHex}`;

        if (!verifyHMAC(encryptedTextForHMAC, hmacFromPayload, HMAC_SECRET_KEY)) {
            throw new Error("HMAC verification failed. QR data integrity compromised.");
        }
        console.log("   HMAC verified successfully.");
        const currentIpfsHistoryCID = decrypt(encryptedTextForHMAC, AES_SECRET_KEY);
        console.log(`   Decrypted current IPFS History CID: ${currentIpfsHistoryCID}`);

        // 2. Download current history from IPFS
        const currentHistoryData = await downloadFromIPFS(currentIpfsHistoryCID);

        // Validate timestamp
        const timestamp = Math.floor(Date.now() / 1000);
        if (currentHistoryData.historyLog && currentHistoryData.historyLog.length > 0) {
            const lastEntry = currentHistoryData.historyLog[currentHistoryData.historyLog.length - 1];
            if (timestamp <= lastEntry.timestamp) {
                throw new Error("New timestamp must be greater than the last history entry");
            }
        }

        // 3. Append new transport event
        console.log("\nAppending new transport event to history...");
        const newHistoryEntry = {
            timestamp: Math.floor(Date.now() / 1000),
            event: "Transport Scan Update",
            actor: transporterWalletAddress,
            location: newLocation,
            details: `Product scanned at new location: ${newLocation} by transporter ${transporterWalletAddress}.`
        };
        const updatedHistoryData = {
            ...currentHistoryData, // Preserve rfid, images, videos etc.
            historyLog: [...(currentHistoryData.historyLog || []), newHistoryEntry]
        };

        // 4. Upload updated history to IPFS
        const newIpfsHistoryCID = await uploadToIPFS(updatedHistoryData, `token_${tokenId}_history_${Date.now()}.json`);

        // 5. Call updateProductHistoryCID on the contract
        console.log("\nUpdating product history CID on the smart contract...");
        const provider = new ethers.JsonRpcProvider(RPC_URL);
        const wallet = new ethers.Wallet(PRIVATE_KEY, provider);
        // Use the ABI of the deployed contract (SupplyChainNFT)
        const supplyChainContract = new ethers.Contract(CONTRACT_ADDRESS, SupplyChainNFT_ABI, wallet);

        console.log(`   Calling updateProductHistoryCID(${tokenId}, ${newIpfsHistoryCID}) on contract ${CONTRACT_ADDRESS}...`);
        // updateProductHistoryCID is in NFTCore, accessible via SupplyChainNFT contract
        const tx = await supplyChainContract.updateProductHistoryCID(tokenId, newIpfsHistoryCID);
        console.log(`   Transaction sent: ${tx.hash}`);
        const receipt = await tx.wait();
        console.log(`   ✅ Transaction confirmed. History CID updated for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);

        console.log("\n✅ Transporter update simulated successfully!");
        console.log(`   New IPFS History CID: ${newIpfsHistoryCID}`);

    } catch (error) {
        console.error("❌ Error simulating transporter update:", error.message);
        if (error.data) console.error("   Error data:", error.data);
        process.exit(1);
    }
}

main();

