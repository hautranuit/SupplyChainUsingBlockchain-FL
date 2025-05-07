// 05_simulate_consumer_view.js
import crypto from "crypto";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import * as Client from "@web3-storage/w3up-client"; // For potential direct IPFS download if needed, though primary is gateway
import { StoreMemory } from "@web3-storage/w3up-client/stores/memory";
import * as Proof from "@web3-storage/w3up-client/proof";
import { Signer } from "@web3-storage/w3up-client/principal/ed25519";

// Determine the correct path to .env file based on script execution location
const __filename_script = fileURLToPath(import.meta.url);
const __dirname_script = path.dirname(__filename_script);
const envPath = path.resolve(__dirname_script, "../ifps_qr.env");
dotenv.config({ path: envPath });

let ipfsClient; // For potential direct download, though gateway is primary

// --- Decryption Helper Functions (from decryptCID.js / 03_simulate_transporter_update.js) ---
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

// --- IPFS Helper Functions (from backendListener.js, slightly adapted) ---
async function initWeb3StorageForDownload() { // Optional, if direct download is preferred over gateway
    const W3UP_KEY = process.env.KEY;
    const W3UP_PROOF = process.env.PROOF;
    if (!W3UP_KEY || !W3UP_PROOF) {
        console.warn("⚠️ Missing Web3.Storage KEY or PROOF. Direct IPFS download via client will not be available.");
        return null;
    }
    try {
        const principal = Signer.parse(W3UP_KEY);
        const store = new StoreMemory();
        const client = await Client.create({ principal, store });
        const proof = await Proof.parse(W3UP_PROOF);
        const space = await client.addSpace(proof);
        await client.setCurrentSpace(space.did());
        console.log("✅ Web3.Storage client initialized for potential direct IPFS download.");
        return client;
    } catch (err) {
        if (err.message.includes("space already registered")) {
            // Attempt to recover by setting current space if already registered
            try {
                const principal = Signer.parse(W3UP_KEY);
                const store = new StoreMemory();
                const client = await Client.create({ principal, store }); // Re-create to list spaces
                const spaces = await client.spaces();
                if (spaces.length > 0) {
                    await client.setCurrentSpace(spaces[0].did());
                    console.log(`✅ Web3.Storage current space set to: ${spaces[0].did()} for potential direct download.`);
                    return client;
                }
            } catch (recoveryErr) {
                console.warn("⚠️ Failed to recover Web3.Storage space for direct download:", recoveryErr.message);
                return null;
            }
        }
        console.warn("⚠️ Error initializing Web3.Storage for direct download:", err.message);
        return null;
    }
}

async function downloadFromIPFSGateway(cid) {
    if (!cid || cid.trim() === "") {
        console.error("❌ Attempted to download from null or empty CID.");
        return null;
    }
    console.log(`[IPFS Gateway] Downloading data from CID: ${cid}`);
    const url = `https://${cid}.ipfs.w3s.link`; // Using w3s.link public gateway
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch from IPFS gateway ${url}: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log("[IPFS Gateway] Data downloaded successfully.");
        return data;
    } catch (error) {
        console.error(`[IPFS Gateway] Error downloading from CID ${cid}:`, error.message);
        return null;
    }
}

async function downloadFromIPFSGatewayWithRetry(cid, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await downloadFromIPFSGateway(cid);
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            console.log(`Retry ${i + 1}/${maxRetries} after error: ${error.message}`);
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
    }
}


async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error("❌ Usage: node 05_simulate_consumer_view.js \"<encryptedQrPayload>\"");
        console.error("Example: node 05_simulate_consumer_view.js \"iv:encrypted:hmac\"");
        process.exit(1);
    }
    const encryptedQrPayload = args[0];

    const AES_SECRET_KEY = process.env.AES_SECRET_KEY;
    const HMAC_SECRET_KEY = process.env.HMAC_SECRET_KEY;

    if (!AES_SECRET_KEY || !HMAC_SECRET_KEY) {
        console.error("❌ Missing AES_SECRET_KEY or HMAC_SECRET_KEY in .env file.");
        process.exit(1);
    }
    if (AES_SECRET_KEY.length !== 64 || HMAC_SECRET_KEY.length !== 64) {
        console.error("❌ AES_SECRET_KEY and HMAC_SECRET_KEY must be 64-character hex strings (32 bytes).");
        process.exit(1);
    }

    try {
        // Initialize IPFS client (optional, for direct download if gateway fails or is not preferred)
        // ipfsClient = await initWeb3StorageForDownload(); 

        // 1. Decrypt QR Payload to get current IPFS History CID
        console.log("\nDecrypting QR payload for consumer view...");
        const parts = encryptedQrPayload.split(":");
        if (parts.length !== 3) throw new Error("Invalid encrypted QR payload format. Expected iv:encrypted:hmac");
        const [ivHex, encryptedHex, hmacFromPayload] = parts;
        const encryptedTextForHMAC = `${ivHex}:${encryptedHex}`;

        if (!verifyHMAC(encryptedTextForHMAC, hmacFromPayload, HMAC_SECRET_KEY)) {
            throw new Error("HMAC verification failed. QR data integrity compromised.");
        }
        console.log("   HMAC verified successfully.");
        const ipfsHistoryCID = decrypt(encryptedTextForHMAC, AES_SECRET_KEY);
        console.log(`   Decrypted IPFS History CID: ${ipfsHistoryCID}`);

        // 2. Download history from IPFS (using public gateway by default)
        console.log("\nFetching product history and data from IPFS...");
        const productData = await downloadFromIPFSGatewayWithRetry(ipfsHistoryCID);

        if (productData) {
            console.log("\n✅ Product Data and History Retrieved Successfully:");
            console.log("---------------------------------------------------");
            console.log("Product RFID Information:");
            console.log(JSON.stringify(productData.rfid || { error: "RFID data not found" }, null, 2));
            console.log("\nProduct Images (CIDs/Links - if available):");
            console.log(JSON.stringify(productData.images || [], null, 2));
            console.log("\nProduct Videos (CIDs/Links - if available):");
            console.log(JSON.stringify(productData.videos || [], null, 2));
            console.log("\nProduct History Log:");
            if (productData.historyLog && productData.historyLog.length > 0) {
                productData.historyLog.forEach(entry => {
                    console.log(`  - Timestamp: ${new Date(entry.timestamp * 1000).toLocaleString()}`);
                    console.log(`    Event: ${entry.event}`);
                    console.log(`    Actor: ${entry.actor}`);
                    if (entry.location) console.log(`    Location: ${entry.location}`);
                    console.log(`    Details: ${entry.details}`);
                    if (entry.transactionHash) console.log(`    Tx Hash: ${entry.transactionHash}`);
                    console.log("---");
                });
            } else {
                console.log("  No history log entries found.");
            }
            console.log("---------------------------------------------------");
        } else {
            console.error("❌ Failed to retrieve product data from IPFS.");
        }

    } catch (error) {
        console.error("❌ Error simulating consumer view:", error.message);
        process.exit(1);
    }
}

main();

