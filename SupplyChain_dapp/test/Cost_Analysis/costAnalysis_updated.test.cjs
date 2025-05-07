const { ethers } = require("hardhat");
const { expect } = require("chai");
// const { performance } = require("perf_hooks"); // Not strictly needed if only measuring gas

// GAS_PRICE_GWEI and TOKEN_PRICE_USD are removed as per requirements
// Traditional system costs are removed as per requirements

describe("Supply Chain Cost Analysis (Batch Processing & IPFS History - Gas Units Only)", function () {
    let supplyChainNFT, owner, manufacturer, t1, t2, t3, t4, t5, retailer, sn1, pn1, pn2, pn3;
    let signers;
    let tokenId;
    const results = { // Store gas costs
        minting: BigInt(0),
        storeCID_initial: BigInt(0),
        scenario_1_transporter: BigInt(0),
        scenario_2_transporters: BigInt(0),
        scenario_3_transporters: BigInt(0),
        scenario_4_transporters: BigInt(0),
        scenario_5_transporters: BigInt(0),
    };

    const UPDATER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("UPDATER_ROLE"));
    const MINTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("MINTER_ROLE"));

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

        let proposeEvent;
        for (const log of receiptPropose.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "BatchProposed") { proposeEvent = parsedLog; break; }
            } catch (error) { /* Ignore */ }
        }
        if (!proposeEvent) throw new Error("BatchProposed event not found.");
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`     - Batch ID: ${batchId}, Selected Validators: ${selectedValidators.length}`);

        // 2. Validate Batch
        let approvals = 0;
        const superMajorityFraction = await supplyChainNFT.superMajorityFraction(); // Fetch from contract
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
        }
        totalGasUsed += validationGas;
        console.log(`     - Validation Gas (Total ${selectedValidators.length} votes): ${validationGas}`);
        console.log(`     - Required ${requiredApprovals} approvals, got ${approvals} approvals.`);

        // 3. Commit Batch
        const commitTx = await supplyChainNFT.connect(proposer).commitBatch(batchId);
        const receiptCommit = await commitTx.wait();
        totalGasUsed += BigInt(receiptCommit.gasUsed);
        console.log(`     - Commit Gas: ${receiptCommit.gasUsed}`);

        expect(await supplyChainNFT.ownerOf(currentTokenId)).to.equal(toNode.address);
        console.log(`     - Transfer Verified.`);

        // 4. Update Product History CID (Simulated IPFS CID update)
        const placeholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        console.log(`     - Updating history with dummy CID: ${placeholderCID}`);
        // Ensure historyUpdater has UPDATER_ROLE
        const updateHistoryTx = await supplyChainNFT.connect(historyUpdater).updateProductHistoryCID(currentTokenId, placeholderCID);
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
        // Needs owner, mfg, retailer, sn1, pn1, pn2, pn3 (7 for this setup)
        if (signers.length < 7) { 
            throw new Error(`Insufficient signers. Need at least 7. Got ${signers.length}.`);
        }
        [owner, manufacturer, retailer, sn1, pn1, pn2, pn3] = signers.slice(0, 7);

        // Assign transporters, reusing some primary/secondary nodes
        t1 = pn1; 
        t2 = pn2; 
        t3 = pn3; 
        t4 = sn1; // sn1 also acts as a transporter
        t5 = retailer; // retailer also acts as a transporter

        const SupplyChainNFTFactory = await ethers.getContractFactory("SupplyChainNFT");
        supplyChainNFT = await SupplyChainNFTFactory.deploy(owner.address);
        await supplyChainNFT.deploymentTransaction().wait(1);
        console.log(`SupplyChainNFT deployed at: ${await supplyChainNFT.getAddress()}`);

        await supplyChainNFT.connect(owner).grantRole(UPDATER_ROLE, sn1.address); // sn1 for history updates
        await supplyChainNFT.connect(owner).grantRole(MINTER_ROLE, owner.address); // owner for minting
        console.log(`Granted UPDATER_ROLE to ${sn1.address}, MINTER_ROLE to ${owner.address}`);

        const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 };
        const ContractNodeType = { Primary: 0, Secondary: 1 };

        console.log("Registering nodes...");
        async function setupNode(signer, role, type, name) {
            await supplyChainNFT.connect(owner).setVerifiedNode(signer.address, true);
            if (role !== null) await supplyChainNFT.connect(owner).setRole(signer.address, role);
            if (type !== null) await supplyChainNFT.connect(owner).setNodeType(signer.address, type);
            console.log(`Registered ${name} (${signer.address}) with Role: ${role}, Type: ${type}`);
        }

        await setupNode(manufacturer, ContractRole.Manufacturer, null, "Manufacturer");
        await setupNode(retailer, ContractRole.Retailer, null, "Retailer (also T5)");
        await setupNode(sn1, null, ContractNodeType.Secondary, "Secondary Node (sn1, Updater, Proposer, T4)");
        await setupNode(pn1, ContractRole.Transporter, ContractNodeType.Primary, "Primary Node (pn1, T1)");
        await setupNode(pn2, ContractRole.Transporter, ContractNodeType.Primary, "Primary Node (pn2, T2)");
        await setupNode(pn3, ContractRole.Transporter, ContractNodeType.Primary, "Primary Node (pn3, T3)");
        // Ensure T4 and T5 also have Transporter role if not covered by pn1-3 setup
        if (t4 !== pn1 && t4 !== pn2 && t4 !== pn3) await setupNode(t4, ContractRole.Transporter, null, "Transporter T4 (sn1)");
        if (t5 !== pn1 && t5 !== pn2 && t5 !== pn3) await setupNode(t5, ContractRole.Transporter, null, "Transporter T5 (retailer)");

        // Mint Product NFT
        console.log("Minting NFT...");
        const mintParams = {
            recipient: manufacturer.address, uniqueProductID: "PROD123", batchNumber: "B001",
            manufacturingDate: "2025-05-01", expirationDate: "2026-05-01", productType: "Electronics",
            manufacturerID: "MANU001", quickAccessURL: "http://example.com/prod123", nftReference: "ipfs://initialhash"
        };
        const txMint = await supplyChainNFT.connect(owner).mintNFT(mintParams);
        const receiptMint = await txMint.wait(1);
        let mintEvent;
        for (const log of receiptMint.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ProductMinted") { mintEvent = parsedLog; break; }
            } catch (error) { /* Ignore */ }
        }
        if (!mintEvent) throw new Error("ProductMinted event not found.");
        tokenId = mintEvent.args.tokenId;
        results.minting = BigInt(receiptMint.gasUsed);
        console.log(`NFT Minted with ID: ${tokenId}. Gas: ${results.minting}`);

        // Store Initial CID
        const initialPlaceholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        console.log(`Storing initial dummy CID: ${initialPlaceholderCID}`);
        const txStoreCID = await supplyChainNFT.connect(owner).storeInitialCID(tokenId, initialPlaceholderCID); // Owner has UPDATER_ROLE (or DEFAULT_ADMIN)
        const receiptStoreCID = await txStoreCID.wait(1);
        results.storeCID_initial = BigInt(receiptStoreCID.gasUsed);
        console.log(`Initial CID stored. Gas: ${results.storeCID_initial}`);
    });

    it("Scenario 1: 1 Transporter (M -> R)", async function () {
        console.log("\nStarting Scenario 1 (1 Transporter)");
        // Proposer: sn1 (Secondary Node), History Updater: sn1 (has UPDATER_ROLE)
        const totalGas = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, retailer, tokenId, "Batch (M -> R)");
        results.scenario_1_transporter = totalGas;
    });

    it("Scenario 2: 2 Transporters (M -> T1 -> R)", async function () {
        console.log("\nStarting Scenario 2 (2 Transporters)");
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, retailer, tokenId, "Batch (T1 -> R)");
        results.scenario_2_transporters = gasStep1 + gasStep2;
    });

    it("Scenario 3: 3 Transporters (M -> T1 -> T2 -> R)", async function () {
        console.log("\nStarting Scenario 3 (3 Transporters)");
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, retailer, tokenId, "Batch (T2 -> R)");
        results.scenario_3_transporters = gasStep1 + gasStep2 + gasStep3;
    });

    it("Scenario 4: 4 Transporters (M -> T1 -> T2 -> T3 -> R)", async function () {
        console.log("\nStarting Scenario 4 (4 Transporters)");
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, t3, tokenId, "Batch (T2 -> T3)");
        const gasStep4 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t3, retailer, tokenId, "Batch (T3 -> R)");
        results.scenario_4_transporters = gasStep1 + gasStep2 + gasStep3 + gasStep4;
    });

    it("Scenario 5: 5 Transporters (M -> T1 -> T2 -> T3 -> T4 -> R)", async function () {
        console.log("\nStarting Scenario 5 (5 Transporters)");
        const gasStep1 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, manufacturer, t1, tokenId, "Batch (M -> T1)");
        const gasStep2 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t1, t2, tokenId, "Batch (T1 -> T2)");
        const gasStep3 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t2, t3, tokenId, "Batch (T2 -> T3)");
        const gasStep4 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t3, t4, tokenId, "Batch (T3 -> T4)");
        const gasStep5 = await executeBatchTransferAndHistoryUpdate(sn1, sn1, t4, retailer, tokenId, "Batch (T4 -> R)");
        results.scenario_5_transporters = gasStep1 + gasStep2 + gasStep3 + gasStep4 + gasStep5;
    });

    after(async function () {
        console.log("\n\n--- Gas Cost Analysis Results (Batch Processing & IPFS History - Gas Units Only) ---");
        console.log("\n--- Initial Costs (One-time per Product) ---");
        console.log(`Minting NFT:                ${results.minting.toString().padStart(12)} Gas`);
        console.log(`Store Initial History CID:  ${results.storeCID_initial.toString().padStart(12)} Gas`);
        const totalInitialGas = results.minting + results.storeCID_initial;
        console.log(`-------------------------------------------------`);
        console.log(`TOTAL INITIAL:              ${totalInitialGas.toString().padStart(12)} Gas`);

        console.log("\n--- Transport Leg Costs (Batch Transfer + History Update per Leg) ---");
        console.log("Scenario (Transporters) | Total Gas Units");
        console.log("------------------------|----------------");
        console.log(`1 Transporter           | ${results.scenario_1_transporter.toString().padStart(14)} Gas`);
        console.log(`2 Transporters          | ${results.scenario_2_transporters.toString().padStart(14)} Gas`);
        console.log(`3 Transporters          | ${results.scenario_3_transporters.toString().padStart(14)} Gas`);
        console.log(`4 Transporters          | ${results.scenario_4_transporters.toString().padStart(14)} Gas`);
        console.log(`5 Transporters          | ${results.scenario_5_transporters.toString().padStart(14)} Gas`);
        
        console.log("\n* Note: Setup gas costs (deployment, role grants, node registration) are excluded from per-scenario results.");
        console.log("* Note: Each scenario's gas includes all batch proposals, validations, commits, and history CID updates for that scenario.");
    });
});

