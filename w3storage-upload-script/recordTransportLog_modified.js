import * as Client from '@web3-storage/w3up-client';
import { StoreMemory } from '@web3-storage/w3up-client/stores/memory';
import * as Proof from '@web3-storage/w3up-client/proof';
import { Signer } from '@web3-storage/w3up-client/principal/ed25519';
import * as dotenv from 'dotenv';
import { ethers } from 'ethers';
const { filesFromPaths } = await import('files-from-path'); // Dynamic import for ES module
import { Blob, File } from '@web-std/file'; // For creating files in memory

dotenv.config();

// --- Configuration (Ensure these are set in your .env file) ---
const RPC_URL = process.ifps_qr.env.AMOY_RPC_URL || "YOUR_AMOY_RPC_URL";
const NFT_CORE_ADDRESS = process.ifps_qr.env.NFT_CORE_ADDRESS || "YOUR_DEPLOYED_NFT_CORE_ADDRESS";
const UPDATER_PRIVATE_KEY = process.ifps_qr.env.BACKEND_PRIVATE_KEY; // Key for account with UPDATER_ROLE
const W3UP_KEY = process.ifps_qr.env.KEY; // Web3.Storage KEY
const W3UP_PROOF = process.ifps_qr.env.PROOF; // Web3.Storage PROOF
const NFTCoreABI = require("../SupplyChain_dapp/artifacts/contracts/NFTCore.sol/NFTCore.json").abi; // Assuming ABI path

// --- Helper Functions ---

// Initialize Web3.Storage client
async function initWeb3Storage() {
    if (!W3UP_KEY || !W3UP_PROOF) {
        throw new Error("Missing Web3.Storage KEY or PROOF in environment variables.");
    }
    const principal = Signer.parse(W3UP_KEY);
    const store = new StoreMemory();
    const client = await Client.create({ principal, store });
    const proof = await Proof.parse(W3UP_PROOF);
    const space = await client.addSpace(proof);
    await client.setCurrentSpace(space.did());
    console.log("Web3.Storage client initialized and space set.");
    return client;
}

// Initialize Smart Contract Interface
function initContract() {
    if (!UPDATER_PRIVATE_KEY) {
        throw new Error("Missing UPDATER_PRIVATE_KEY (BACKEND_PRIVATE_KEY) in environment variables.");
    }
    if (!NFT_CORE_ADDRESS || NFT_CORE_ADDRESS === "YOUR_DEPLOYED_NFT_CORE_ADDRESS") {
        throw new Error("NFT_CORE_ADDRESS is not set or is a placeholder.");
    }
    const provider = new ethers.JsonRpcProvider(RPC_URL);
    const wallet = new ethers.Wallet(UPDATER_PRIVATE_KEY, provider);
    const contract = new ethers.Contract(NFT_CORE_ADDRESS, NFTCoreABI, wallet);
    console.log(`Contract interface initialized for address: ${NFT_CORE_ADDRESS}`);
    return contract;
}

// Download data from IPFS (using fetch and gateway for simplicity)
async function downloadFromIPFS(cid) {
    if (!cid) {
        console.warn("Attempted to download from null or empty CID. Returning empty history.");
        return { historyLog: [] }; // Return a default structure if no previous CID
    }
    const url = `https://${cid}.ipfs.w3s.link`; // Use w3s link gateway
    console.log(`Attempting to download previous history from: ${url}`);
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch from IPFS gateway: ${response.statusText}`);
        }
        const data = await response.json();
        console.log("Successfully downloaded previous history.");
        return data;
    } catch (error) {
        console.error(`Error downloading from IPFS (CID: ${cid}):`, error);
        // Decide how to handle download errors - perhaps return empty history or rethrow
        console.warn("Returning empty history due to download error.");
        return { historyLog: [] };
    }
}

// Upload data to IPFS
async function uploadToIPFS(client, data) {
    console.log("Preparing data for IPFS upload...");
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const file = new File([blob], 'product_history.json');
    console.log("Uploading updated history file to Web3.Storage...");
    const cid = await client.uploadFile(file);
    console.log(`Upload complete. New CID: ${cid}`);
    return cid.toString();
}

// --- Main Function ---
async function main() {
    const tokenId = process.argv[2];
    const location = process.argv[3];
    const staffWallet = process.argv[4];
    const previousCID = process.argv[5]; // This CID comes from the DECRYPTED QR code payload

    if (!tokenId || !location || !staffWallet) { // previousCID can be null/empty initially
        console.error("❌ Usage: node recordTransportLog.js <tokenId> <location> <staffWallet> [previousCID]");
        console.error("   Note: previousCID is obtained from the decrypted QR code.");
        process.exit(1);
    }

    console.log(`
--- Recording Transport Log ---`);
    console.log(`   Token ID: ${tokenId}`);
    console.log(`   Location: ${location}`);
    console.log(`   Staff Wallet: ${staffWallet}`);
    console.log(`   Previous History CID (from QR): ${previousCID || 'None'}`);

    try {
        // 1. Initialize clients
        const ipfsClient = await initWeb3Storage();
        const contract = initContract();

        // 2. Download previous history from IPFS using the CID from QR
        //    Important: The QR code should contain the CID of the *last known history state*.
        const previousHistoryData = await downloadFromIPFS(previousCID);

        // 3. Create new history entry
        const newHistoryEntry = {
            timestamp: Math.floor(Date.now() / 1000),
            event: "Transport Scan",
            actor: staffWallet,
            location: location,
            details: `Product scanned at location ${location} by staff ${staffWallet}.`
            // You could add more details like GPS coordinates if available
        };

        // 4. Prepare updated history data (keeping existing RFID, images etc. if they are in the root)
        const updatedHistoryData = {
            ...previousHistoryData, // Preserve other top-level fields if any (like rfid, images)
            historyLog: [
                ...(previousHistoryData.historyLog || []), // Keep previous logs
                newHistoryEntry
            ]
        };

        // 5. Upload updated history to IPFS
        const newCID = await uploadToIPFS(ipfsClient, updatedHistoryData);

        // 6. Update smart contract with the new CID (requires UPDATER_ROLE)
        console.log(`⏳ Calling updateProductHistory(${tokenId}, ${newCID}) on contract...`);
        const tx = await contract.updateProductHistory(tokenId, newCID);
        console.log(`   Transaction sent: ${tx.hash}`);
        const receipt = await tx.wait();
        console.log(`✅ Transaction confirmed. History updated on-chain for token ${tokenId}. Gas used: ${receipt.gasUsed.toString()}`);
        console.log(`--- Transport Log Recorded Successfully ---`);

    } catch (err) {
        console.error("❌ Error during transport log recording:", err);
        process.exit(1);
    }
}

main();

