// 04_generate_updated_qr.js
import { ethers } from "ethers";
import QRCode from "qrcode";
import crypto from "crypto";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { cleanupOldQRCodes } from './utils.js';

// Determine the correct path to .env file and artifact based on script execution location
const __filename_script = fileURLToPath(import.meta.url);
const __dirname_script = path.dirname(__filename_script);
const envPath = path.resolve(__dirname_script, "../ifps_qr.env");
dotenv.config({ path: envPath });

// Load SupplyChainNFT.json manually - THIS IS THE DEPLOYED CONTRACT
const supplyChainNFTArtifactPath = path.resolve(__dirname_script, "../../SupplyChain_dapp/artifacts/contracts/SupplyChainNFT.sol/SupplyChainNFT.json");
const SupplyChainNFTArtifact = JSON.parse(fs.readFileSync(supplyChainNFTArtifactPath, "utf-8"));
const SupplyChainNFT_ABI = SupplyChainNFTArtifact.abi;

// AES encryption (AES-256-CBC with IV)
function encrypt(text, secretKey) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv("aes-256-cbc", Buffer.from(secretKey, "hex"), iv);
    let encrypted = cipher.update(text, "utf8", "hex");
    encrypted += cipher.final("hex");
    return iv.toString("hex") + ":" + encrypted;
}

// Generate HMAC for integrity check
function generateHMAC(encryptedText, hmacKey) {
    return crypto.createHmac("sha256", Buffer.from(hmacKey, "hex")).update(encryptedText).digest("hex");
}

// Save QR code as PNG
async function generateQRCodeToFile(finalPayload, outputPath) {
    try {
        await QRCode.toFile(outputPath, finalPayload, {
            color: { dark: "#000", light: "#fff" },
            width: 300
        });
        console.log(`✅ Encrypted QR code saved to: ${outputPath}`);
    } catch (err) {
        console.error("❌ Error generating QR code image:", err);
        throw err;
    }
}

async function main() {
    const args = process.argv.slice(2);
    if (args.length < 1) {
        console.error("❌ Usage: node 04_generate_updated_qr.js <tokenId>");
        process.exit(1);
    }
    const tokenId = args[0];

    const RPC_URL = process.env.POLYGON_AMOY_RPC || process.env.AMOY_RPC_URL;
    const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS; // This should be the deployed SupplyChainNFT address
    const AES_SECRET_KEY = process.env.AES_SECRET_KEY;
    const HMAC_SECRET_KEY = process.env.HMAC_SECRET_KEY;

    if (!RPC_URL || !CONTRACT_ADDRESS || !AES_SECRET_KEY || !HMAC_SECRET_KEY) {
        console.error("❌ Missing RPC_URL, CONTRACT_ADDRESS, AES_SECRET_KEY, or HMAC_SECRET_KEY in .env file.");
        process.exit(1);
    }
    if (AES_SECRET_KEY.length !== 64 || HMAC_SECRET_KEY.length !== 64) {
        console.error("❌ AES_SECRET_KEY and HMAC_SECRET_KEY must be 64-character hex strings (32 bytes).");
        process.exit(1);
    }

    const provider = new ethers.JsonRpcProvider(RPC_URL);
    // Use the ABI of the deployed contract (SupplyChainNFT)
    const supplyChainContract = new ethers.Contract(CONTRACT_ADDRESS, SupplyChainNFT_ABI, provider);

    console.log(`Fetching LATEST IPFS CID for Token ID: ${tokenId} from contract ${CONTRACT_ADDRESS}...`);
    console.log("(This should reflect any recent history updates)");

    try {
        // cidMapping is in NFTCore, accessible via SupplyChainNFT contract
        const ipfsCID = await supplyChainContract.cidMapping(tokenId);
        if (!ipfsCID || ipfsCID.trim() === "") {
            console.error(`❌ No IPFS CID found for Token ID: ${tokenId}. Ensure the product has been minted and history updated.`);
            process.exit(1);
        }
        console.log(`   Found LATEST IPFS CID: ${ipfsCID}`);

        console.log("\nEncrypting IPFS CID...");
        const encryptedCID = encrypt(ipfsCID, AES_SECRET_KEY);
        console.log(`   Encrypted data (iv:ciphertext): ${encryptedCID}`);

        console.log("Generating HMAC...");
        const hmac = generateHMAC(encryptedCID, HMAC_SECRET_KEY);
        console.log(`   HMAC: ${hmac}`);

        const finalPayload = `${encryptedCID}:${hmac}`;
        console.log(`\nFinal QR Payload for updated history: ${finalPayload}`);

        const qrOutputDir = path.resolve(__dirname_script, "../qr_codes");
        if (!fs.existsSync(qrOutputDir)) {
            fs.mkdirSync(qrOutputDir, { recursive: true });
        }
        
        // Clean up old QR codes BEFORE creating new one
        cleanupOldQRCodes(tokenId);
        
        const qrFilePath = path.join(qrOutputDir, `token_${tokenId}_updated_qr_${Date.now()}.png`);
        await generateQRCodeToFile(finalPayload, qrFilePath);

        console.log("\n✅ Updated QR Code generated successfully.");
        console.log(`   QR Image Path: ${qrFilePath}`);
        console.log(`   Encrypted Payload for next step (consumer view): ${finalPayload}`);

    } catch (error) {
        console.error("❌ Error generating updated QR code:", error.message);
        if (error.data) console.error("   Error data:", error.data);
        process.exit(1);
    }
}

main();

