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

    const ARBITRATOR_ROLE = ethers.keccak256(ethers.toUtf8Bytes("ARBITRATOR_ROLE"));
    const ContractRole = { Manufacturer: 0, Transpoter: 1, Customer: 2, Arbitrator: 3 }; // Corrected to match contract enum
    const ContractNodeType = { Primary: 0, Secondary: 1 };

    // Setup runs once before all tests in this script
    before(async function () {
        signers = await ethers.getSigners();
        console.log(`Number of available signers: ${signers.length}`);
        // Need owner, mfg, buyer, 2 arbitrators, 1 voter = 6 signers
        if (signers.length < 6) {
            console.error("ERROR: Insufficient signers. This script requires at least 6 signers.");
            throw new Error(`Insufficient signers available. Need at least 6, but got ${signers.length}.`);
        }
        // Assign roles (6 signers)
        [owner, manufacturer, buyer, arbitrator1, arbitrator2, voter1] = signers.slice(0, 6);

        console.log("\n--- Starting Initial Setup for Dispute Tests --- ");
        let tx, receipt;

        // 1. Deploy Contract (Gas cost NOT recorded)
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

        // 2. Register Nodes (including arbitrators)
        console.log("Registering Nodes...");
        async function registerNode(addr, isVerified, role, nodeType, name) {
            console.log(`  Registering ${name} (${addr})...`);
            if (isVerified !== null) await (await supplyChainNFT.connect(owner).setVerifiedNode(addr, isVerified)).wait(1);
            if (role !== null) await (await supplyChainNFT.connect(owner).setRole(addr, role)).wait(1);
            if (nodeType !== null) await (await supplyChainNFT.connect(owner).setNodeType(addr, nodeType)).wait(1);
        }

        await registerNode(manufacturer.address, true, ContractRole.Manufacturer, null, "Manufacturer");
        // Removed retailer registration as role doesn't exist in contract
        await registerNode(buyer.address, true, ContractRole.Customer, null, "Buyer");
        await registerNode(arbitrator1.address, true, ContractRole.Arbitrator, null, "Arbitrator 1");
        await registerNode(arbitrator2.address, true, ContractRole.Arbitrator, null, "Arbitrator 2");
        // Register voter just as verified node
        await registerNode(voter1.address, true, null, null, "Voter 1");
        console.log("Node Registration Complete.");

        // 3. Mint Product NFT
        console.log("Minting NFT...");
        const mintParams = { recipient: manufacturer.address, uniqueProductID: "PROD-DISPUTE-TEST", batchNumber: "B-DISPUTE", manufacturingDate: "2025-05-01", expirationDate: "2026-05-01", productType: "DisputeProduct", manufacturerID: "MANU-DISPUTE", quickAccessURL: "http://example.com/dispute", nftReference: "ipfs://nftDisputeHash" };
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

        // 4. Store Initial CID (Simulated)
        initialPlaceholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        console.log(`Storing initial dummy CID: ${initialPlaceholderCID}`);
        await (await supplyChainNFT.connect(owner).storeCID(tokenId, initialPlaceholderCID)).wait(1);
        console.log("Initial CID stored.");

        // 5. Simulate a Purchase to set up for potential dispute
        console.log("Simulating Purchase...");
        const price = ethers.parseEther("0.01");
        await (await supplyChainNFT.connect(manufacturer).sellProduct(tokenId, price)).wait(1);
        await (await supplyChainNFT.connect(buyer).initiatePurchase(tokenId, initialPlaceholderCID)).wait(1);
        tx = await supplyChainNFT.connect(buyer).depositPurchaseCollateral(tokenId, { value: price });
        receipt = await tx.wait(1);
        disputeBlockNumber = receipt.blockNumber; // Store block number after purchase/transfer
        console.log(`Purchase simulated. NFT transferred to buyer ${buyer.address} in block ${disputeBlockNumber}.`);
        expect(await supplyChainNFT.ownerOf(tokenId)).to.equal(buyer.address);

        console.log("--- Initial Setup Complete ---");
    });

    // --- Dispute Scenarios --- 

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

        // 1. Open Dispute (Buyer opens dispute)
        console.log(`  Opening dispute with ${candidates.length} candidates...`);
        tx = await supplyChainNFT.connect(buyer).openDispute(tokenId, candidateAddresses);
        receipt = await tx.wait(1);
        scenarioGas.openDispute = BigInt(receipt.gasUsed);
        console.log(`    - openDispute Gas: ${scenarioGas.openDispute}`);

        // 2. Vote for Arbitrator
        console.log(`  Casting ${votes.length} votes...`);
        for (let i = 0; i < votes.length; i++) {
            const vote = votes[i];
            tx = await supplyChainNFT.connect(vote.voter).voteForArbitrator(tokenId, vote.candidate.address);
            receipt = await tx.wait(1);
            scenarioGas.voteForArbitrator_total += BigInt(receipt.gasUsed);
        }
        console.log(`    - voteForArbitrator (Total for ${votes.length} votes) Gas: ${scenarioGas.voteForArbitrator_total}`);

        // 3. Select Arbitrator (Anyone can call, using owner here)
        console.log(`  Selecting arbitrator...`);
        tx = await supplyChainNFT.connect(owner).selectArbitrator(tokenId);
        receipt = await tx.wait(1);
        scenarioGas.selectArbitrator = BigInt(receipt.gasUsed);
        const selected = await supplyChainNFT.selectedArbitrator(tokenId);
        console.log(`    - selectArbitrator Gas: ${scenarioGas.selectArbitrator}. Selected: ${selected}`);

        // Find the signer object for the selected arbitrator
        const selectedArbitratorSigner = signers.find(s => s.address === selected);
        if (!selectedArbitratorSigner) throw new Error(`Signer for selected arbitrator ${selected} not found.`);

        // 4. Resolve Dispute (Selected arbitrator resolves)
        console.log(`  Resolving dispute by selected arbitrator ${selected}...`);
        const decision = true; // Arbitrator decides in favor (example)
        tx = await supplyChainNFT.connect(selectedArbitratorSigner).resolveDispute(tokenId, disputeBlockNumber, decision);
        receipt = await tx.wait(1);
        scenarioGas.resolveDispute = BigInt(receipt.gasUsed);
        console.log(`    - resolveDispute Gas: ${scenarioGas.resolveDispute}`);

        scenarioGas.total = scenarioGas.openDispute + scenarioGas.voteForArbitrator_total + scenarioGas.selectArbitrator + scenarioGas.resolveDispute;
        console.log(`  -> Total Gas for Scenario ${scenarioId}: ${scenarioGas.total}`);
        results.scenario = { id: scenarioId, candidates: candidates.length, votes: votes.length, gas: scenarioGas };
        
        // Reset dispute state for next scenario if needed (re-minting might be cleaner)
        // For now, assume each scenario runs independently on the same tokenId state after setup.
        // If state needs reset, might need beforeEach instead of before, or manual state reset.
        // Let's mark as resolved to prevent re-opening in subsequent tests on same token.
        // await supplyChainNFT.connect(owner). // Need a way to reset disputeResolved[tokenId] for testing, maybe add admin function?
        // For now, we will let tests fail if state isn't reset. A better approach would use beforeEach.
    }

    it("Scenario: Dispute with 2 Candidates, 1 Vote", async function () {
        const candidates = [arbitrator1, arbitrator2]; // Use the 2 available arbitrators
        const votes = [{ voter: voter1, candidate: arbitrator1 }]; // Use the 1 available voter
        await runDisputeScenario(1, candidates, votes);
    });

    // Removed Scenarios 2-5

    // --- Final Report (Gas Units Only) ---
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
        console.log("* Note: This script requires 7 signers.");
    });
});

