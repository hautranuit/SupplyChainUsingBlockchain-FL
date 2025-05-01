import * as Client from '@web3-storage/w3up-client';
import { StoreMemory } from '@web3-storage/w3up-client/stores/memory';
import * as Proof from '@web3-storage/w3up-client/proof';
import { Signer } from '@web3-storage/w3up-client/principal/ed25519';
import * as dotenv from 'dotenv';
import { createWriteStream, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { ethers } from 'ethers';
import { finished } from 'stream/promises';
const { filesFromPaths } = await import('files-from-path');

dotenv.config();

const abi = [
    "function updateProductHistory(uint256 tokenId, string memory cid) public"
];

// === Init Web3.Storage client ===
async function initWeb3Storage() {
    const principal = Signer.parse(process.env.KEY);
    const store = new StoreMemory();
    const client = await Client.create({ principal, store });
    const proof = await Proof.parse(process.env.PROOF);
    const space = await client.addSpace(proof);
    await client.setCurrentSpace(space.did());
    return client;
}

// === Init smart contract ===
function initContract() {
    const rpcUrl = process.env.POLYGON_AMOY_RPC;
    const privateKey = process.env.PRIVATE_KEY;
    const contractAddress = ethers.getAddress(process.env.CONTRACT_ADDRESS);
    const provider = new ethers.JsonRpcProvider(rpcUrl);
    const wallet = new ethers.Wallet(privateKey, provider);
    const contract = new ethers.Contract(contractAddress, abi, wallet);
    return contract;
}

// === Format timestamp for filename ===
function getFormattedTimestampFilename() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(now.getHours())}h${pad(now.getMinutes())}_${pad(now.getDate())}_${pad(now.getMonth() + 1)}_${now.getFullYear()}`;
}

// === Save and upload product history ===
async function saveAndUploadEventData(client, dataContent, tokenId, outputDir) {
    const formattedTimestamp = getFormattedTimestampFilename();
    const fileName = `ProductHistory_${formattedTimestamp}.txt`;
    const filePath = join(outputDir, fileName);

    const fileStream = createWriteStream(filePath);
    fileStream.write(dataContent);
    fileStream.end();
    await finished(fileStream);
    console.log(`📄 Product history saved to: ${filePath}`);

    const files = await filesFromPaths([filePath]);
    const cid = await client.uploadDirectory(files);
    const cidStr = cid.toString();

    console.log(`✅ Uploaded to IPFS: https://${cidStr}.ipfs.w3s.link/${fileName}`);
    return cidStr;
}

// === Main Function ===
async function main() {
    const tokenId = process.argv[2];
    const location = process.argv[3];
    const staffWallet = process.argv[4];
    const previousCID = process.argv[5]; // decrypted from QR

    if (!tokenId || !location || !staffWallet || !previousCID) {
        console.error("❌ Usage: node recordTransportLog.js <tokenId> <location> <staffWallet> <previousCID>");
        process.exit(1);
    }

    try {
        const client = await initWeb3Storage();
        const contract = initContract();

        const now = new Date();
        const timestampStr = now.toISOString();
        const message = `The product has been shipped via location ${location}.\nStaff: ${staffWallet}\nTimestamp: ${timestampStr}\nPrevious CID: ${previousCID}`;
        
        const currentDir = dirname(fileURLToPath(import.meta.url));
        const outputDir = join(currentDir, 'output');
        mkdirSync(outputDir, { recursive: true });

        const newCID = await saveAndUploadEventData(client, message, tokenId, outputDir);

        console.log(`⏳ Updating smart contract with new history CID for token ${tokenId}...`);
        const tx = await contract.updateProductHistory(tokenId, newCID);
        await tx.wait();
        console.log(`✅ Updated on-chain! 🔗 TxHash: ${tx.hash}`);

    } catch (err) {
        console.error("❌ Error:", err.message);
    }
}

main();
