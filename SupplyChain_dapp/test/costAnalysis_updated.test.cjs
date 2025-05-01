const { ethers } = require("hardhat");
const { expect } = require("chai");
const { performance } = require("perf_hooks");

// --- Configuration Placeholders ---
const GAS_PRICE_GWEI = 0.06; 
const GAS_PRICE_WEI = BigInt(GAS_PRICE_GWEI * 1e9); // Calculate Wei price 
const TOKEN_PRICE_USD = 0.2351; 
// ----------------------------------

// --- Traditional System Costs ---
const traditionalCosts = {
    "1": 1.70,
    "2": 2.68,
    "3": 3.82,
    "4": 5.15,
    "5": 6.50,
};
// ---------------------------------

describe("Supply Chain Cost Analysis (with IPFS History Costs)", function () {
    let supplyChainNFT, owner, manufacturer, t1, t2, t3, t4, t5, retailer, sn1, pn1, pn2, pn3, pn4, pn5;
    let signers;
    let tokenId;
    const results = { // Store gas costs
        minting: BigInt(0),
        storeCID_initial: BigInt(0),
        "1": BigInt(0),
        "2": BigInt(0),
        "3": BigInt(0),
        "4": BigInt(0),
        "5": BigInt(0),
    };

    // Roles Enum (assuming similar structure in NodeManagement)
    const Role = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, SecondaryNode: 4, PrimaryNode: 5 }; // Adjust as per your contract
    const UPDATER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("UPDATER_ROLE"));
    const DEFAULT_ADMIN_ROLE = "0x0000000000000000000000000000000000000000000000000000000000000000";

    // Helper function to execute a full batch cycle + history update
    async function executeBatchTransferAndHistoryUpdate(proposer, historyUpdater, fromNode, toNode, currentTokenId, batchDesc) {
        console.log(`   Executing ${batchDesc}: ${fromNode.address} -> ${toNode.address}`);
        let totalGasUsed = BigInt(0);
        let validationGas = BigInt(0);

        // 1. Propose Batch
        const txData = [{ from: fromNode.address, to: toNode.address, tokenId: currentTokenId }];
        const proposeTx = await supplyChainNFT.connect(proposer).proposeBatch(txData);
        const receiptPropose = await proposeTx.wait();
        totalGasUsed += BigInt(receiptPropose.gasUsed);
        console.log(`     - Propose Gas: ${receiptPropose.gasUsed}`);

        // Robust event parsing for BatchProposed
        let proposeEvent;
        for (const log of receiptPropose.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "BatchProposed") {
                    proposeEvent = parsedLog;
                    break;
                }
            } catch (error) {
                // Ignore logs that don't match the contract's interface
            }
        }

        if (!proposeEvent) throw new Error("BatchProposed event not found in transaction logs.");
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`     - Batch ID: ${batchId}, Selected Validators: ${selectedValidators.length}`);

        // 2. Validate Batch (Ensure Super-Majority Approve)
        const validatorVotes = {};
        let approvals = 0;
        const superMajorityFraction = await supplyChainNFT.superMajorityFraction();
        const requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajorityFraction) / 100);

        for (let i = 0; i < selectedValidators.length; i++) {
            const validatorAddr = selectedValidators[i];
            const validatorSigner = signers.find(s => s.address === validatorAddr);
            if (!validatorSigner) {
                console.warn(`     - Could not find signer for selected validator ${validatorAddr}`);
                continue;
            }
            let vote = approvals < requiredApprovals;
            if (vote) approvals++;

            const validateTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote);
            const receiptValidate = await validateTx.wait();
            validationGas += BigInt(receiptValidate.gasUsed);
            validatorVotes[validatorAddr] = vote;
        }
        totalGasUsed += validationGas;
        console.log(`     - Validation Gas (Total ${selectedValidators.length} votes): ${validationGas}`);
        console.log(`     - Required ${requiredApprovals} approvals, got ${approvals} approvals.`);

        // 3. Commit Batch
        const commitTx = await supplyChainNFT.connect(proposer).commitBatch(batchId);
        const receiptCommit = await commitTx.wait();
        totalGasUsed += BigInt(receiptCommit.gasUsed);
        console.log(`     - Commit Gas: ${receiptCommit.gasUsed}`);

        // Verify transfer
        expect(await supplyChainNFT.ownerOf(currentTokenId)).to.equal(toNode.address);
        console.log(`     - Transfer Verified.`);

        // 4. Update Product History (Simulated IPFS CID update)
        const placeholderCID = "bafkrei" + Math.random().toString(36).substring(2); // Generate a dummy CID
        console.log(`     - Updating history with dummy CID: ${placeholderCID}`);
        const updateHistoryTx = await supplyChainNFT.connect(historyUpdater).updateProductHistory(currentTokenId, placeholderCID);
        const receiptUpdateHistory = await updateHistoryTx.wait();
        const updateHistoryGas = BigInt(receiptUpdateHistory.gasUsed);
        totalGasUsed += updateHistoryGas;
        console.log(`     - Update History Gas: ${updateHistoryGas}`);

        console.log(`     - Total Gas for this step (incl. history update): ${totalGasUsed}`);
        return totalGasUsed;
    }

    beforeEach(async function () {
        signers = await ethers.getSigners();
        console.log(`Number of available signers: ${signers.length}`);
        if (signers.length < 7) { // Reduced requirement to 7
            throw new Error(`Insufficient signers available. Need at least 7, but got ${signers.length}. Check Hardhat network configuration.`);
        }
        // Assign roles using available signers
        [owner, manufacturer, retailer, sn1, pn1, pn2, pn3] = signers.slice(0, 7);

        // Reuse signers for transporters (assign to outer scope variables)
        t1 = pn1; // Example: Reuse pn1 as t1
        t2 = pn2; // Example: Reuse pn2 as t2
        t3 = pn3; // Example: Reuse pn3 as t3
        t4 = sn1; // Example: Reuse sn1 as t4
        t5 = retailer; // Example: Reuse retailer as t5 (Note: Potential conflict in Scenario 5)

        const SupplyChainNFTFactory = await ethers.getContractFactory("SupplyChainNFT");
        // Deploy with owner as initial admin
        supplyChainNFT = await SupplyChainNFTFactory.deploy(owner.address);
        // Wait for the deployment transaction to be mined
        await supplyChainNFT.deploymentTransaction().wait(1);

        // Grant UPDATER_ROLE to sn1 (representing the backend/authorized updater)
        let tx = await supplyChainNFT.connect(owner).grantRole(UPDATER_ROLE, sn1.address);
        await tx.wait(1); // Wait for confirmation
        console.log(`Granted UPDATER_ROLE to ${sn1.address}`);

        // Define Enums based on NodeManagement.sol
        const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 }; // Adjusted based on NodeManagement
        const ContractNodeType = { Primary: 0, Secondary: 1 };

        // Register Nodes using setVerifiedNode, setNodeType, setRole
        console.log("Registering nodes...");

        // Manufacturer
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(manufacturer.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(manufacturer.address, ContractRole.Manufacturer);
        await tx.wait(1);
        // NodeType not explicitly defined for Manufacturer, assuming default or not applicable

        // Retailer
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(retailer.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(retailer.address, ContractRole.Retailer);
        await tx.wait(1);
        // NodeType not explicitly defined for Retailer

        // Secondary Node (sn1) - Also T4 and History Updater
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(sn1.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setNodeType(sn1.address, ContractNodeType.Secondary);
        await tx.wait(1);
        // Role for Secondary Node not explicitly defined in NodeManagement enum, might need adjustment

        // Primary Nodes (pn1, pn2, pn3) - Also T1, T2, T3
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(pn1.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setNodeType(pn1.address, ContractNodeType.Primary);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(pn2.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setNodeType(pn2.address, ContractNodeType.Primary);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(pn3.address, true);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setNodeType(pn3.address, ContractNodeType.Primary);
        await tx.wait(1);

        // Register reused signers as Transporters (T1-T5)
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(t1.address, true); // pn1 as T1
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(t1.address, ContractRole.Transporter);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(t2.address, true); // pn2 as T2
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(t2.address, ContractRole.Transporter);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(t3.address, true); // pn3 as T3
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(t3.address, ContractRole.Transporter);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(t4.address, true); // sn1 as T4
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(t4.address, ContractRole.Transporter);
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setVerifiedNode(t5.address, true); // retailer as T5
        await tx.wait(1);
        tx = await supplyChainNFT.connect(owner).setRole(t5.address, ContractRole.Transporter);
        await tx.wait(1);

        // Mint Product NFT
        console.log("Minting NFT...");
        const mintParams = {
            recipient: manufacturer.address,
            uniqueProductID: "PROD123",
            batchNumber: "B001",
            manufacturingDate: "2025-05-01",
            expirationDate: "2026-05-01",
            productType: "Electronics",
            manufacturerID: "aaaaa",
            quickAccessURL: "http://example.com/prod123",
            nftReference: "ipfs://somehash"
        };
        // Use owner (admin) to mint
        tx = await supplyChainNFT.connect(owner).mintNFT(mintParams);
        const receiptMint = await tx.wait(1); // Wait for confirmation

        // Robust event parsing for ProductMinted
        let mintEvent;
        for (const log of receiptMint.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ProductMinted") {
                    mintEvent = parsedLog;
                    break;
                }
            } catch (error) {
                // Ignore logs that don't match the contract's interface
            }
        }

        if (!mintEvent) {
            throw new Error("ProductMinted event not found in transaction logs.");
        }

        tokenId = mintEvent.args.tokenId;
        console.log(`NFT Minted with ID: ${tokenId}`);
        results.minting = BigInt(receiptMint.gasUsed);

        // Store Initial CID (Simulated)
        const initialPlaceholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        console.log(`Storing initial dummy CID: ${initialPlaceholderCID}`);
        // Use owner (admin) to store initial CID
        tx = await supplyChainNFT.connect(owner).storeCID(tokenId, initialPlaceholderCID);
        const receiptStoreCid = await tx.wait(1); // Wait for confirmation
        results.storeCID_initial = BigInt(receiptStoreCid.gasUsed);
        console.log(`Initial CID stored. Gas: ${results.storeCID_initial}`);
    });

    it("Scenario 1: 50-100 miles, 1 Transporter", async function () {
        console.log("\nStarting Scenario 1 (1 Transporter)");
        // M -> R (1 batch transfer + 1 history update)
        // Proposer: sn1, History Updater: sn1
        const totalGas = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, retailer, tokenId, "Batch 1 (M -> R)");
        results["1"] = totalGas;
    });

    it("Scenario 2: 100-250 miles, 2 Transporters", async function () {
        console.log("\nStarting Scenario 2 (2 Transporters)");
        // M -> T1 -> R (2 batch transfers + 2 history updates)
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch 1 (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, retailer, tokenId, "Batch 2 (T1 -> R)");
        results["2"] = gasStep1 + gasStep2;
    });

    it("Scenario 3: 250-500 miles, 3 Transporters", async function () {
        console.log("\nStarting Scenario 3 (3 Transporters)");
        // M -> T1 -> T2 -> R (3 batch transfers + 3 history updates)
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch 1 (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch 2 (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, retailer, tokenId, "Batch 3 (T2 -> R)");
        results["3"] = gasStep1 + gasStep2 + gasStep3;
    });

    it("Scenario 4: 500-750 miles, 4 Transporters", async function () {
        console.log("\nStarting Scenario 4 (4 Transporters)");
        // M -> T1 -> T2 -> T3 -> R (4 batch transfers + 4 history updates)
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch 1 (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch 2 (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, t3, tokenId, "Batch 3 (T2 -> T3)");
        const gasStep4 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t3, retailer, tokenId, "Batch 4 (T3 -> R)");
        results["4"] = gasStep1 + gasStep2 + gasStep3 + gasStep4;
    });

    it("Scenario 5: 750-1000 miles, 5 Transporters", async function () {
        console.log("\nStarting Scenario 5 (5 Transporters)");
        // M -> T1 -> T2 -> T3 -> T4 -> R (5 batch transfers + 5 history updates)
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch 1 (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch 2 (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, t3, tokenId, "Batch 3 (T2 -> T3)");
        const gasStep4 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t3, t4, tokenId, "Batch 4 (T3 -> T4)");
        const gasStep5 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t4, retailer, tokenId, "Batch 5 (T4 -> R)");
        results["5"] = gasStep1 + gasStep2 + gasStep3 + gasStep4 + gasStep5;
    });

    after(async function () {
        console.log("\n\n--- Cost Analysis Results (Including IPFS History Updates) ---");
        console.log(`Gas Price Used: ${GAS_PRICE_GWEI} Gwei`);
        console.log(`Token Price Used: $${TOKEN_PRICE_USD.toFixed(2)} USD/Token`);
        console.log("-----------------------------------------------------------------------------------------------------------------");
        console.log("| Scenario | Distance (miles) | Transporters | Traditional Cost (USD) | Blockchain Gas Cost (Total) | Blockchain Cost (USD) | Cost Reduction (%) |");
        console.log("|----------|------------------|--------------|--------------------------|-----------------------------|-------------------------|--------------------|");

        const scenarios = [
            { scenario: "1", distance: "50-100", transporters: 1 },
            { scenario: "2", distance: "100-250", transporters: 2 },
            { scenario: "3", distance: "250-500", transporters: 3 },
            { scenario: "4", distance: "500-750", transporters: 4 },
            { scenario: "5", distance: "750-1000", transporters: 5 },
        ];

        for (const sc of scenarios) {
            const scenarioId = sc.scenario;
            // Total gas includes batch transfers + history updates for the scenario
            const totalGas = results[scenarioId] || BigInt(0);
            const traditionalCost = traditionalCosts[scenarioId] || 0;

            const costWei = totalGas * GAS_PRICE_WEI;
            const costToken = parseFloat(ethers.formatUnits(costWei, 'wei'));
            const costUsd = costToken * TOKEN_PRICE_USD;

            let costReduction = 0;
            if (traditionalCost > 0) {
                costReduction = ((traditionalCost - costUsd) / traditionalCost) * 100;
            }

            console.log(
                `| ${scenarioId.padEnd(8)} | ${sc.distance.padEnd(16)} | ${String(sc.transporters).padEnd(12)} | ${('$' + traditionalCost.toFixed(2)).padEnd(24)} | ${totalGas.toString().padEnd(27)} | ${('$' + costUsd.toFixed(2)).padEnd(23)} | ${costReduction.toFixed(2).padStart(18)}% |`);
        }
        console.log("-----------------------------------------------------------------------------------------------------------------");
        console.log("* Blockchain Gas Cost (Total) includes proposeBatch, validateBatch, commitBatch, and updateProductHistory for each transport leg.");
        console.log("* Blockchain Cost (USD) is an estimate based on the placeholder GAS_PRICE_GWEI and TOKEN_PRICE_USD.");
        console.log("* Update these constants with current market values for accurate USD cost estimation.");
        console.log(`* One-time Costs (not included in scenario totals): Minting Gas = ${results.minting}, Initial storeCID Gas = ${results.storeCID_initial}`);
    });
});

