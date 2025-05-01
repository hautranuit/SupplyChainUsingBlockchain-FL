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
// Dynamic import for files-from-path
const { filesFromPaths } = await import('files-from-path');

dotenv.config();

const abi = [
    // Events & Functions from NFTCore and Marketplace
    "event ProductMinted(address indexed owner, uint256 tokenId)",
    "event ProductHistoryUpdated(uint256 tokenId, string history)",
    "event SaleSuccessful(uint256 tokenId, address buyer, uint256 price)",
    "function rfidDataMapping(uint256 tokenId) view returns (string uniqueProductID, string batchNumber, string manufacturingDate, string expirationDate, string productType, string manufacturerID, string quickAccessURL, string nftReference)",
    "function storeCID(uint256 tokenId, string memory cid) public",
    "function updateProductHistory(uint256 tokenId, string memory cid) public",
];


async function initWeb3Storage() {
    if (!process.env.KEY || !process.env.PROOF) {
        throw new Error('Missing KEY or PROOF in .env file');
    }
    const principal = Signer.parse(process.env.KEY);
    const store = new StoreMemory();
    const client = await Client.create({ principal, store });
    const proof = await Proof.parse(process.env.PROOF);
    if (!proof) {
        throw new Error('Failed to extract delegation proof');
    }
    const space = await client.addSpace(proof);
    await client.setCurrentSpace(space.did());
    return client;
}

function initContract() {
    const rpcUrl = process.env.POLYGON_AMOY_RPC;
    const privateKey = process.env.PRIVATE_KEY;
    const contractAddress = ethers.getAddress(process.env.CONTRACT_ADDRESS);

    if (!rpcUrl || !privateKey || !contractAddress) {
        throw new Error('Missing POLYGON_AMOY_RPC, PRIVATE_KEY, or CONTRACT_ADDRESS in .env file');
    }
    const provider = new ethers.JsonRpcProvider(rpcUrl);
    const wallet = new ethers.Wallet(privateKey, provider);
    const contract = new ethers.Contract(contractAddress, abi, wallet);
    return contract;
}

// Function to format timestamp for filename
function getFormattedTimestampFilename() {
    const currentDate = new Date();
    const hours = String(currentDate.getHours()).padStart(2, '0');
    const minutes = String(currentDate.getMinutes()).padStart(2, '0');
    const day = String(currentDate.getDate()).padStart(2, '0');
    const month = String(currentDate.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
    const year = currentDate.getFullYear();
    // Format: HHhMM_DD_MM_YYYY
    return `${hours}h${minutes}_${day}_${month}_${year}`;
}

// Helper function to save and upload event data
async function saveAndUploadEventData(client, dataContent, eventBaseName, outputDir) {
    const formattedTimestamp = getFormattedTimestampFilename();
    // Filename will be eventBaseName (always 'ProductHistory' as per new requirement) + timestamp
    const fileName = `${eventBaseName}_${formattedTimestamp}.txt`;
    const filePath = join(outputDir, fileName);

    const fileStream = createWriteStream(filePath);
    fileStream.write(dataContent);
    fileStream.end();
    await finished(fileStream);
    console.log(`   📄 Event data has been written to: ${filePath}`);

    console.log(`   ⏳ Uploading ${fileName} to IPFS...`); // Log the actual filename
    const filesToUpload = [filePath];
    const dataFiles = await filesFromPaths(filesToUpload);
    const dataFilesCid = await client.uploadDirectory(dataFiles);
    const cidString = dataFilesCid.toString();

    console.log(`   ✅ Upload of ${fileName} complete!`);
    console.log(`   🔗 IPFS Link: https://${cidString}.ipfs.w3s.link/${fileName}`);

    return cidString;
}

async function main() {
    try {
        const client = await initWeb3Storage();
        const contract = initContract();

        const currentDir = dirname(fileURLToPath(import.meta.url));
        const outputDir = join(currentDir, 'output');
        mkdirSync(outputDir, { recursive: true });

        console.log("Listening for events from the smart contract...");

        // --- Listener for ProductMinted (RFID filename format remains unchanged) ---
        contract.on("ProductMinted", async (owner, tokenId, event) => {
            console.log(`\n🎉 ProductMinted event detected!`);
            console.log(`   Owner: ${owner}`);
            console.log(`   Token ID: ${tokenId.toString()}`);
            // ... (RFID handling part remains unchanged) ...
            try {
                const rfidData = await contract.rfidDataMapping(tokenId);
                const [ uniqueProductID, batchNumber, manufacturingDate, expirationDate, productType, manufacturerID, quickAccessURL, nftReference ] = rfidData;
                // Content labels are already in English in the original structure
                const rfidContent = `Unique Product ID: ${uniqueProductID}\nBatch Number: ${batchNumber}\nManufacturing Date: ${manufacturingDate}\nExpiration Date: ${expirationDate}\nProduct Type: ${productType}\nManufacturer ID: ${manufacturerID}\nQuick Access URL: ${quickAccessURL}\nNFT Reference: ${nftReference}`.trim();
                const rfidFileName = `rfidData_token${tokenId.toString()}.txt`;
                const rfidFilePath = join(outputDir, rfidFileName);
                const rfidFileStream = createWriteStream(rfidFilePath);
                rfidFileStream.write(rfidContent);
                rfidFileStream.end();
                console.log(`   📄 RFID data has been written to: ${rfidFilePath}`);

                const imagePath = join(currentDir, 'images', 'product1.png');
                const videoPath = join(currentDir, 'videos', 'product1.mp4');
                const filesToUpload = [rfidFilePath, imagePath, videoPath];
                console.log('   ⏳ Uploading RFID, image, and video to IPFS...');
                const allFiles = await filesFromPaths(filesToUpload);
                const allFilesCid = await client.uploadDirectory(allFiles);
                const cidString = allFilesCid.toString();
                console.log('   ✅ Upload complete!');
                console.log(`   🔗 RFID Data: https://${cidString}.ipfs.w3s.link/${rfidFileName}`);
                console.log(`   🔗 Product Image: https://${cidString}.ipfs.w3s.link/${imagePath.split('/').pop()}`);
                console.log(`   🔗 Product Video: https://${cidString}.ipfs.w3s.link/${videoPath.split('/').pop()}`);

                console.log(`   ⏳ Storing CID ${cidString} on the blockchain...`);
                const transaction = await contract.storeCID(tokenId, cidString);
                await transaction.wait();
                console.log(`   ✅ CID stored on the blockchain.`);
                console.log(`   ⛓️ Transaction Hash: ${transaction.hash}`);
            } catch (error) {
                 console.error(`   ❌ Error processing ProductMinted for token ${tokenId.toString()}:`, error);
            }
        });

        // --- Listener for ProductHistoryUpdated (Uses filename "ProductHistory_...") ---
        contract.on("ProductHistoryUpdated", async (tokenId, history, event) => {
            console.log(`\n🔄 ProductHistoryUpdated event detected!`);
            console.log(`   Token ID: ${tokenId.toString()}`);
            console.log(`   History: ${history}`);
            try {
                const now = new Date();
                const timeStampStr = now.toLocaleTimeString('en-GB') + ', ' + now.toLocaleDateString('en-GB'); 

                const historyWithTimestamp = `${history}\n\n[Timestamp: ${timeStampStr}]`;
                // Call helper function, eventBaseName is always 'ProductHistory'
                const historyCidString = await saveAndUploadEventData(client, historyWithTimestamp, 'ProductHistory', outputDir);

                // Update history CID on the blockchain
                console.log(`   ⏳ Updating History CID ${historyCidString} on the blockchain for token ${tokenId.toString()}...`);
                const updateTransaction = await contract.updateProductHistory(tokenId, historyCidString);
                await updateTransaction.wait();
                console.log(`   ✅ History CID updated on the blockchain.`);
                console.log(`   ⛓️ Transaction Hash: ${updateTransaction.hash}`);
            } catch (error) {
                console.error(`   ❌ Error processing ProductHistoryUpdated for token ${tokenId.toString()}:`, error);
            }
        });

    } catch (error) {
        console.error('❌ An error occurred during initialization or listening:', error);
    }
}

main();