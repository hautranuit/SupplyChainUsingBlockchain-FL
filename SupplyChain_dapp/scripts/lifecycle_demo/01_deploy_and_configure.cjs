const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Helper function for delays
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    const signers = await ethers.getSigners();
    const deployer = signers[0]; // Will also be the initial admin for roles
    const manufacturerAcc = signers[1];
    const transporter1Acc = signers[2]; // Primary Node
    const transporter2Acc = signers[3]; // Primary Node
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
        // For local testing, you might not need these or can use lower values.
    };

    // Adding a small delay before deployment
    await delay(500); 

    console.log("Attempting deployment with gasOptions:", gasOptions);
    const supplyChainNFT = await SupplyChainNFTFactory.deploy(deployer.address, gasOptions);
    await supplyChainNFT.waitForDeployment();
    const contractAddress = await supplyChainNFT.getAddress();
    console.log("SupplyChainNFT deployed to:", contractAddress);
    console.log("Transaction hash:", supplyChainNFT.deploymentTransaction().hash);

    console.log("\n--- Initial Configuration --- ");

    const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 };
    const ContractNodeType = { Primary: 0, Secondary: 1 };

    console.log("\\\\n--- Granting Contract-Level Roles ---");
    const MINTER_ROLE = await supplyChainNFT.MINTER_ROLE();
    const UPDATER_ROLE = await supplyChainNFT.UPDATER_ROLE();

    console.log(`Granting MINTER_ROLE to deployer (${deployer.address})...`);
    await (await supplyChainNFT.connect(deployer).grantRole(MINTER_ROLE, deployer.address, gasOptions)).wait();
    console.log(`Granting UPDATER_ROLE to deployer (${deployer.address})...`);
    await (await supplyChainNFT.connect(deployer).grantRole(UPDATER_ROLE, deployer.address, gasOptions)).wait();

    console.log("\\\\n--- Configuring Demo Participants ---");

    async function configureParticipant(account, name, role, nodeType, initialReputation, isVerified = true) {
        console.log(`Configuring ${name} (${account.address})...`);
        if (isVerified) {
            await (await supplyChainNFT.connect(deployer).setVerifiedNode(account.address, true, gasOptions)).wait();
            console.log(`  - Set as Verified Node`);
        }
        if (role !== null) {
            await (await supplyChainNFT.connect(deployer).setRole(account.address, role, gasOptions)).wait();
            console.log(`  - Set Role to: ${Object.keys(ContractRole).find(key => ContractRole[key] === role)} (${role})`);
        }
        if (nodeType !== null) {
            await (await supplyChainNFT.connect(deployer).setNodeType(account.address, nodeType, gasOptions)).wait();
            console.log(`  - Set Node Type to: ${Object.keys(ContractNodeType).find(key => ContractNodeType[key] === nodeType)} (${nodeType})`);
        }
        if (initialReputation > 0) {
            await (await supplyChainNFT.connect(deployer).adminUpdateReputation(account.address, initialReputation, gasOptions)).wait();
            console.log(`  - Set Initial Reputation to: ${initialReputation}`);
        }
    }

    await configureParticipant(manufacturerAcc, "Manufacturer", ContractRole.Manufacturer, ContractNodeType.Primary, 100);
    await configureParticipant(transporter1Acc, "Transporter 1 (Secondary)", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(transporter2Acc, "Transporter 2 (Secondary)", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(transporter3Acc, "Transporter 3 (Secondary)", ContractRole.Transporter, ContractNodeType.Secondary, 100);
    await configureParticipant(retailerAcc, "Retailer (Primary)", ContractRole.Retailer, ContractNodeType.Primary, 100);
    await configureParticipant(buyerAcc, "Buyer/Customer (Secondary)", ContractRole.Customer, ContractNodeType.Secondary, 100);
    await configureParticipant(arbitratorAcc, "Arbitrator (Primary)", ContractRole.Arbitrator, ContractNodeType.Primary, 100);

    console.log("\\n--- Configuration Summary --- ");
    console.log("Contract Address:", contractAddress);
    console.log("Deployer/Admin:", deployer.address);
    console.log("Manufacturer:", manufacturerAcc.address);
    // ... (log other participants)

    // Save deployment info to demo_context.json
    const deploymentInfo = {
        contractAddress: contractAddress,
        deployerAddress: deployer.address,
        manufacturerAddress: manufacturerAcc.address,
        transporter1Address: transporter1Acc.address,
        transporter2Address: transporter2Acc.address,
        transporter3Address: transporter3Acc.address,
        retailerAddress: retailerAcc.address,
        buyer1Address: buyerAcc.address,
        arbitratorAddress: arbitratorAcc.address
    };

    // Save to the lifecycle_demo directory only
    const contextPath = path.join(__dirname, "demo_context.json");

    try {
        // Create directory if it doesn't exist (though __dirname should exist)
        const dir = path.dirname(contextPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        fs.writeFileSync(contextPath, JSON.stringify(deploymentInfo, null, 2));
        console.log("\\nDeployment info saved to:", contextPath);
    } catch (error) {
        console.error(`Warning: Could not save to ${contextPath}:`, error.message);
    }

    // Update .env file
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
        // Filter out empty lines that might result from multiple newlines at the end
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

