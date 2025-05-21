require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

console.log("=== Environment Variables Debug ===");
console.log("Current working directory:", process.cwd());
console.log("Current file path:", __filename);

const fs = require('fs');
const path = require('path');
const projectRoot = path.resolve(__dirname);
const envPath = path.join(projectRoot, '.env');

console.log("Project root:", projectRoot);
console.log("Looking for .env file at:", envPath);
console.log(".env file exists:", fs.existsSync(envPath));

if (fs.existsSync(envPath)) {
    console.log(".env file content length:", fs.statSync(envPath).size);
    const envContent = fs.readFileSync(envPath, 'utf8');
    console.log("First few characters of .env:", envContent.substring(0, 50) + "...");
}

require('dotenv').config({ path: envPath });

console.log("After reloading .env:");
console.log("POLYGON_AMOY_RPC:", process.env.POLYGON_AMOY_RPC ? "defined" : "undefined");
console.log("PRIVATE_KEYS:", process.env.PRIVATE_KEYS ? "defined" : "undefined");

task("run-scenario03", "Runs scenario 03 with specified token IDs and their corresponding IPFS CIDs for metadata verification.")
    .addParam("tokenid1", "Token ID for product 1")
    .addParam("cid1", "IPFS CID of metadata for product 1 (from cidMapping)")
    .addParam("tokenid2", "Token ID for product 2")
    .addParam("cid2", "IPFS CID of metadata for product 2 (from cidMapping)")
    .addParam("tokenid3", "Token ID for product 3")
    .addParam("cid3", "IPFS CID of metadata for product 3 (from cidMapping)")
    .setAction(async (taskArgs, hre) => {
        console.log("Task arguments received:", taskArgs);
        
        process.env.SCENARIO03_TOKENID1 = taskArgs.tokenid1;
        process.env.SCENARIO03_CID1 = taskArgs.cid1;
        process.env.SCENARIO03_TOKENID2 = taskArgs.tokenid2;
        process.env.SCENARIO03_CID2 = taskArgs.cid2;
        process.env.SCENARIO03_TOKENID3 = taskArgs.tokenid3;
        process.env.SCENARIO03_CID3 = taskArgs.cid3;
        
        console.log("Running 03_scenario_marketplace_and_purchase.cjs via hre.run...");
        const scriptPath = path.join(__dirname, "scripts", "lifecycle_demo", "03_scenario_marketplace_and_purchase.cjs");
        console.log(`Attempting to run script at absolute path: ${scriptPath}`);
        await hre.run("run", { script: scriptPath, noCompile: true });
        
        console.log("Task run-scenario03 finished.");
    });

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
    solidity: {
        version: "0.8.28", 
        settings: {
            optimizer: {
                enabled: true,
                runs: 1000,
            },
            viaIR: true 
        },
    },
    networks: {
        amoy: {
            url: process.env.POLYGON_AMOY_RPC,
            accounts: process.env.PRIVATE_KEYS ? process.env.PRIVATE_KEYS.split(",").map(key => `0x${key.trim()}`) : [],
            chainId: 80002,
            gas: 30_000_000, // This is likely gasLimit
            maxPriorityFeePerGas: 40000000000, // 35 Gwei in wei
            maxFeePerGas: 70000000000, // 70 Gwei in wei
        },
    },
    mocha: { 
        timeout: 600000 
    }
};
