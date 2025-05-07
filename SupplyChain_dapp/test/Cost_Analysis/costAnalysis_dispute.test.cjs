const { ethers } = require("hardhat");
const { expect } = require("chai");

describe("Supply Chain Cost Analysis (Dispute Resolution Only)", function () {
    let supplyChainNFT, owner, manufacturer, buyer, arbitrator1, arbitrator2, voter1; // Required 6 signers
    let signers;
    let tokenId;
    let initialPlaceholderCID;
    let disputeBlockNumber; // To store block number when dispute might occur

    // Store gas costs for the single dispute scenario
    const results = {
        scenario: null, // Object to hold results for the single scenario
    };

    // const ARBITRATOR_ROLE = ethers.keccak256(ethers.toUtf8Bytes("ARBITRATOR_ROLE")); // Not used directly
    const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 }; // Matches contract enum
    // const ContractNodeType = { Primary: 0, Secondary: 1 }; // Not used in this test

    // Setup runs once before all tests in this script
    before(async function () {
        signers = await ethers.getSigners();
        console.log(`Number of available signers: ${signers.length}`);
        if (signers.length < 6) {
            console.error("ERROR: Insufficient signers. This script requires at least 6 signers.");
            throw new Error(`Insufficient signers available. Need at least 6, but got ${signers.length}.`);
        }
        [owner, manufacturer, buyer, arbitrator1, arbitrator2, voter1] = signers.slice(0, 6);

        console.log("\n--- Starting Initial Setup for Dispute Tests --- ");
        let tx, receipt;

        console.log("Deploying SupplyChainNFT...");
        const SupplyChainNFTFactory = await ethers.getContractFactory("SupplyChainNFT");
        try {
            supplyChainNFT = await SupplyChainNFTFactory.deploy(owner.address);
            await supplyChainNFT.deploymentTransaction().wait(1);
            console.log(`Contract Deployed at ${await supplyChainNFT.getAddress()}`);
        } catch (error) {
            console.error("ERROR: Contract deployment failed.", error);
            this.skip();
            return;
        }

        console.log("Registering Nodes...");
        async function registerNode(addr, isVerified, role, nodeType, name) {
            console.log(`  Registering ${name} (${addr})...`);
            if (isVerified !== null) await (await supplyChainNFT.connect(owner).setVerifiedNode(addr, isVerified)).wait(1);
            if (role !== null) await (await supplyChainNFT.connect(owner).setRole(addr, role)).wait(1);
            // NodeType not set for these roles in this test based on original
        }

        await registerNode(manufacturer.address, true, ContractRole.Manufacturer, null, "Manufacturer");
        await registerNode(buyer.address, true, ContractRole.Customer, null, "Buyer");
        await registerNode(arbitrator1.address, true, ContractRole.Arbitrator, null, "Arbitrator 1");
        await registerNode(arbitrator2.address, true, ContractRole.Arbitrator, null, "Arbitrator 2");
        await registerNode(voter1.address, true, null, null, "Voter 1");
        console.log("Node Registration Complete.");

        console.log("Minting NFT...");
        const mintParams = { 
            recipient: manufacturer.address, 
            uniqueProductID: "PROD-DISPUTE-TEST", 
            batchNumber: "B-DISPUTE", 
            manufacturingDate: "2025-05-01", 
            expirationDate: "2026-05-01", 
            productType: "DisputeProduct", 
            manufacturerID: "MANU-DISPUTE", 
            quickAccessURL: "http://example.com/dispute", 
            nftReference: "ipfs://nftDisputeHash" 
        };
        // Assuming owner has MINTER_ROLE or is admin
        tx = await supplyChainNFT.connect(owner).mintNFT(mintParams);
        receipt = await tx.wait(1);
        let mintEvent;
        for (const log of receipt.logs) {
            try {
                const parsedLog = supplyChainNFT.interface.parseLog(log);
                if (parsedLog && parsedLog.name === "ProductMinted") { mintEvent = parsedLog; break; }
            } catch (error) { /* Ignore */ }
        }
        if (!mintEvent) throw new Error("ProductMinted event not found.");
        tokenId = mintEvent.args.tokenId;
        console.log(`NFT Minted with ID: ${tokenId}.`);

        console.log("Storing initial dummy CID...");
        initialPlaceholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        // Assuming owner has UPDATER_ROLE or is admin for storeInitialCID
        await (await supplyChainNFT.connect(owner).storeInitialCID(tokenId, initialPlaceholderCID)).wait(1);
        console.log(`Initial CID stored: ${initialPlaceholderCID}`);

        console.log("Simulating Purchase...");
        const price = ethers.parseEther("0.01");
        await (await supplyChainNFT.connect(manufacturer).sellProduct(tokenId, price)).wait(1);
        await (await supplyChainNFT.connect(buyer).initiatePurchase(tokenId, initialPlaceholderCID)).wait(1);
        tx = await supplyChainNFT.connect(buyer).depositPurchaseCollateral(tokenId, { value: price });
        receipt = await tx.wait(1);
        disputeBlockNumber = receipt.blockNumber;
        console.log(`Purchase simulated. NFT transferred to buyer ${buyer.address} in block ${disputeBlockNumber}.`);
        expect(await supplyChainNFT.ownerOf(tokenId)).to.equal(buyer.address);

        console.log("--- Initial Setup Complete ---");
    });

    async function runDisputeScenario(scenarioId, candidates, votes) {
        console.log(`\n--- Running Dispute Scenario ${scenarioId} ---`);
        let scenarioGas = {
            openDispute: BigInt(0),
            voteForArbitrator_total: BigInt(0),
            selectArbitrator: BigInt(0),
            resolveDispute: BigInt(0),
            total: BigInt(0)
        };
        let tx, receipt;
        const candidateAddresses = candidates.map(c => c.address);

        console.log(`  Opening dispute with ${candidates.length} candidates...`);
        tx = await supplyChainNFT.connect(buyer).openDispute(tokenId, candidateAddresses);
        receipt = await tx.wait(1);
        scenarioGas.openDispute = BigInt(receipt.gasUsed);
        console.log(`    - openDispute Gas: ${scenarioGas.openDispute}`);

        console.log(`  Casting ${votes.length} votes...`);
        for (let i = 0; i < votes.length; i++) {
            const vote = votes[i];
            tx = await supplyChainNFT.connect(vote.voter).voteForArbitrator(tokenId, vote.candidate.address);
            receipt = await tx.wait(1);
            scenarioGas.voteForArbitrator_total += BigInt(receipt.gasUsed);
        }
        console.log(`    - voteForArbitrator (Total for ${votes.length} votes) Gas: ${scenarioGas.voteForArbitrator_total}`);

        console.log(`  Selecting arbitrator...`);
        tx = await supplyChainNFT.connect(owner).selectArbitrator(tokenId);
        receipt = await tx.wait(1);
        scenarioGas.selectArbitrator = BigInt(receipt.gasUsed);
        const selected = await supplyChainNFT.selectedArbitrator(tokenId);
        console.log(`    - selectArbitrator Gas: ${scenarioGas.selectArbitrator}. Selected: ${selected}`);

        const selectedArbitratorSigner = signers.find(s => s.address === selected);
        if (!selectedArbitratorSigner) throw new Error(`Signer for selected arbitrator ${selected} not found.`);

        console.log(`  Resolving dispute by selected arbitrator ${selected}...`);
        const decision = true; // Arbitrator decides in favor (example)
        tx = await supplyChainNFT.connect(selectedArbitratorSigner).resolveDispute(tokenId, disputeBlockNumber, decision);
        receipt = await tx.wait(1);
        scenarioGas.resolveDispute = BigInt(receipt.gasUsed);
        console.log(`    - resolveDispute Gas: ${scenarioGas.resolveDispute}`);

        scenarioGas.total = scenarioGas.openDispute + scenarioGas.voteForArbitrator_total + scenarioGas.selectArbitrator + scenarioGas.resolveDispute;
        console.log(`  -> Total Gas for Scenario ${scenarioId}: ${scenarioGas.total}`);
        results.scenario = { id: scenarioId, candidates: candidates.length, votes: votes.length, gas: scenarioGas };
    }

    it("Scenario: Dispute with 2 Candidates, 1 Vote", async function () {
        const candidates = [arbitrator1, arbitrator2];
        const votes = [{ voter: voter1, candidate: arbitrator1 }];
        await runDisputeScenario(1, candidates, votes);
    });

    after(async function () {
        console.log("\n\n--- Gas Cost Analysis Results (Dispute Resolution Only - Gas Units) ---");
        if (!results.scenario) {
            console.log("The dispute scenario did not run successfully.");
            return;
        }
        console.log("\nScenario | Candidates | Votes | openDispute Gas | voteForArbitrator (Total) Gas | selectArbitrator Gas | resolveDispute Gas | TOTAL GAS");
        console.log("---------|------------|-------|-----------------|-------------------------------|----------------------|--------------------|-----------");
        const scenario = results.scenario;
        const gas = scenario.gas;
        console.log(
            `${scenario.id.toString().padStart(8)} | ` +
            `${scenario.candidates.toString().padStart(10)} | ` +
            `${scenario.votes.toString().padStart(5)} | ` +
            `${gas.openDispute.toString().padStart(15)} | ` +
            `${gas.voteForArbitrator_total.toString().padStart(29)} | ` +
            `${gas.selectArbitrator.toString().padStart(20)} | ` +
            `${gas.resolveDispute.toString().padStart(18)} | ` +
            `${gas.total.toString().padStart(9)}`
        );
        console.log("\n* Note: Setup gas costs (deployment, registration, minting, purchase) are excluded.");
        console.log("* Note: voteForArbitrator gas is the total for all votes cast in the scenario.");
        console.log("* Note: This script requires 6 signers as configured in the setup.");
    });
});

