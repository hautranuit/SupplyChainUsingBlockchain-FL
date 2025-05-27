const { ethers } = require("hardhat");
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

// --- Helper Function Definition ---
const contextFilePath = path.join(__dirname, 'demo_context.json'); // Centralized data file for FL

function readAndUpdateContext(updateFn) {
    let contextData = {};
    if (fs.existsSync(contextFilePath)) {
        try {
            const fileContent = fs.readFileSync(contextFilePath, 'utf8');
            // Handle empty file case
            if (fileContent.trim() === "") {
                contextData = {};
            } else {
                contextData = JSON.parse(fileContent);
            }
        } catch (error) {
            console.error(`Error reading or parsing ${contextFilePath}:`, error);
            // Start fresh if parsing fails or file is corrupted
            contextData = {};
        }
    }

    // Apply the update function to modify the context data
    const updatedContext = updateFn(contextData);

    try {
        fs.writeFileSync(contextFilePath, JSON.stringify(updatedContext, null, 2));
        console.log(`Context data updated successfully in ${contextFilePath}`);
    } catch (error) {
        console.error(`Error writing context data to ${contextFilePath}:`, error);
    }
    return updatedContext; // Return the updated data for potential immediate use
}

// Helper function for delays
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
// --- End Helper Function ---

async function main() {
    const signers = await ethers.getSigners();
    const deployer = signers[0]; // Will also be the initial admin for roles
    const manufacturerAcc = signers[1];
    const transporter1Acc = signers[2]; 
    const transporter2Acc = signers[3]; 
    const transporter3Acc = signers[4]; // Secondary Node, also batch proposer
    const retailerAcc = signers[5];
    const buyerAcc = signers[6];
    const arbitratorAcc = signers[7];

    if (signers.length < 8) {
        console.error("This script requires at least 8 signers for deployer, manufacturer, 3 transporters, retailer, buyer, and arbitrator.");
        process.exit(1);
    }

    console.log("Deploying SupplyChainNFT with account:", deployer.address);
    const SupplyChainNFTFactory = await ethers.getContractFactory("SupplyChainNFT", deployer);

    // Define gas options to override defaults if needed, especially for testnets like Amoy
    const gasOptions = {
        maxPriorityFeePerGas: ethers.parseUnits('30', 'gwei'), // Adjusted to meet network demands
        maxFeePerGas: ethers.parseUnits('100', 'gwei')       // Adjusted to meet network demands
    };

    // Adding a small delay before deployment
    await delay(500);

    console.log("Attempting deployment..."); 
    const supplyChainNFT = await SupplyChainNFTFactory.deploy(deployer.address, gasOptions);
    await supplyChainNFT.waitForDeployment();
    const contractAddress = await supplyChainNFT.getAddress();
    console.log("SupplyChainNFT deployed to:", contractAddress);
    console.log("Transaction hash:", supplyChainNFT.deploymentTransaction().hash);

    console.log("\n--- Initial Configuration --- ");

    const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 };
    const ContractNodeType = { Primary: 0, Secondary: 1 };
    const RoleNames = { 0: "Manufacturer", 1: "Transporter", 2: "Customer", 3: "Retailer", 4: "Arbitrator" };
    const NodeTypeNames = { 0: "Primary", 1: "Secondary" };

    console.log("\n--- Granting Contract-Level Roles ---");
    const MINTER_ROLE = await supplyChainNFT.MINTER_ROLE();
    const UPDATER_ROLE = await supplyChainNFT.UPDATER_ROLE();

    console.log(`Granting MINTER_ROLE to deployer (${deployer.address})...`);
    await (await supplyChainNFT.connect(deployer).grantRole(MINTER_ROLE, deployer.address, gasOptions)).wait();
    console.log(`Granting UPDATER_ROLE to deployer (${deployer.address})...`);
    await (await supplyChainNFT.connect(deployer).grantRole(UPDATER_ROLE, deployer.address, gasOptions)).wait();

    console.log("\n--- Configuring Demo Participants ---");

    // Store participant details temporarily
    const participants = [];

    async function configureParticipant(account, name, role, nodeType, initialReputation, isVerified = true) {
        console.log(`Configuring ${name} (${account.address})...`);
        const txReceipts = {};
        if (isVerified) {
            const tx = await supplyChainNFT.connect(deployer).setVerifiedNode(account.address, true, gasOptions);
            txReceipts.verify = await tx.wait();
            console.log(`  - Set as Verified Node (Gas: ${txReceipts.verify.gasUsed.toString()})`);
        }
        if (role !== null) {
            const tx = await supplyChainNFT.connect(deployer).setRole(account.address, role, gasOptions);
            txReceipts.role = await tx.wait();
            console.log(`  - Set Role to: ${Object.keys(ContractRole).find(key => ContractRole[key] === role)} (${role}) (Gas: ${txReceipts.role.gasUsed.toString()})`);
        }
        if (nodeType !== null) {
            const tx = await supplyChainNFT.connect(deployer).setNodeType(account.address, nodeType, gasOptions);
            txReceipts.nodeType = await tx.wait();
            console.log(`  - Set Node Type to: ${Object.keys(ContractNodeType).find(key => ContractNodeType[key] === nodeType)} (${nodeType}) (Gas: ${txReceipts.nodeType.gasUsed.toString()})`);
        }
        if (initialReputation > 0) {
            const tx = await supplyChainNFT.connect(deployer).adminUpdateReputation(account.address, initialReputation, gasOptions);
            txReceipts.reputation = await tx.wait();
            console.log(`  - Set Initial Reputation to: ${initialReputation} (Gas: ${txReceipts.reputation.gasUsed.toString()})`);
        }

        // Store details for context_data.json
        participants.push({
            address: account.address.toLowerCase(), // Use lowercase for consistency
            name: name,
            role: role, // Keep numeric for potential processing
            roleName: RoleNames[role] || "Unknown",
            nodeType: nodeType, // Keep numeric
            nodeTypeName: NodeTypeNames[nodeType] || "Unknown",
            initialReputation: initialReputation,
            isVerified: isVerified,
            txReceipts: txReceipts // Store transaction receipts for potential analysis
        });
    }

    // Configure all participants
    await configureParticipant(manufacturerAcc, "Manufacturer", ContractRole.Manufacturer, ContractNodeType.Primary, 100);
    await configureParticipant(transporter1Acc, "Transporter 1", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(transporter2Acc, "Transporter 2", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(transporter3Acc, "Transporter 3 (Batch Proposer)", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(retailerAcc, "Retailer", ContractRole.Retailer, ContractNodeType.Primary, 100);
    await configureParticipant(buyerAcc, "Buyer/Customer", ContractRole.Customer, ContractNodeType.Secondary, 100);
    await configureParticipant(arbitratorAcc, "Arbitrator", ContractRole.Arbitrator, ContractNodeType.Primary, 100);

    // Add deployer/admin as well
    participants.push({
        address: deployer.address.toLowerCase(),
        name: "Deployer/Admin",
        role: null, 
        roleName: "Admin",
        nodeType: null,
        nodeTypeName: "Admin",
        initialReputation: 0, 
        isVerified: true,
        txReceipts: {} // No specific config transactions for deployer itself here
    });


    console.log("\n--- Saving Initial Context Data --- ");

    // Use the helper function to update context_data.json
    readAndUpdateContext(currentContext => {
        // Initialize or update the context structure
        const newContext = {
            contractAddress: contractAddress.toLowerCase(),
            nodes: {}, // Initialize nodes object
            products: {}, // Initialize products object
            batches: {}, // Initialize batches object
            disputes: {} // Initialize disputes object
        };

        // Populate the nodes object
        participants.forEach(p => {
            newContext.nodes[p.address] = {
                address: p.address,
                name: p.name,
                role: p.role,
                roleName: p.roleName,
                nodeType: p.nodeType,
                nodeTypeName: p.nodeTypeName,
                initialReputation: p.initialReputation,
                currentReputation: p.initialReputation, // Start with initial
                isVerified: p.isVerified,
                interactions: [], // Initialize interactions array
                // Optionally add tx hashes/gas used from p.txReceipts if needed for analysis
                // configTxHashes: Object.values(p.txReceipts).map(r => r.transactionHash)
            };
        });

        return newContext;
    });

    // Update .env file (Kept the original logic)
    const envFilePath = path.join(__dirname, "../../../w3storage-upload-script/ifps_qr.env");
    console.log(`\n--- Updating .env file at ${envFilePath} ---`);
    try {
        let envContent = "";
        if (fs.existsSync(envFilePath)) {
            envContent = fs.readFileSync(envFilePath, "utf8");
        }
        const lines = envContent.split("\n");
        let found = false;
        const newLines = lines.map(line => {
            if (line.startsWith("CONTRACT_ADDRESS=")) {
                found = true;
                return `CONTRACT_ADDRESS=${contractAddress.toLowerCase()}`;
            }
            return line;
        });
        if (!found) {
            newLines.push(`CONTRACT_ADDRESS=${contractAddress.toLowerCase()}`);
        }
        fs.writeFileSync(envFilePath, newLines.filter(line => line.trim() !== "" || newLines.indexOf(line) === newLines.length -1 && line === "").join("\n"));
        console.log(`Successfully updated CONTRACT_ADDRESS in ${envFilePath}`);
    } catch (error) {
        console.error(`Error updating ${envFilePath}:`, error);
        console.warn("Please update the CONTRACT_ADDRESS manually in your .env file.");
    }

    console.log("\nDeployment and initial configuration complete.");
    console.log(`Deployed Contract Address: ${contractAddress}`);

}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });

