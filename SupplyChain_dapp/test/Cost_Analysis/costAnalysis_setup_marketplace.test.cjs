const { ethers } = require("hardhat");
const { expect } = require("chai");

// --- Configuration (No USD calculations needed) ---

describe("Supply Chain Cost Analysis (Setup & Marketplace Only)", function () {
    let supplyChainNFT, owner, manufacturer, retailer, sn1, pn1, pn2, buyer; // Reduced signers based on usage
    let signers;
    let tokenId;
    let initialPlaceholderCID;

    // Store gas costs for setup and marketplace phases
    const results = {
        grantRole_total: BigInt(0),
        nodeRegistration_total: BigInt(0), // Overall total for all node reg steps
        minting: BigInt(0),
        storeCID_initial: BigInt(0),
        sellProduct: BigInt(0),
        initiatePurchase: BigInt(0),
        depositCollateral: BigInt(0),
        releasePayment: BigInt(0),
        // Individual registration step totals (for detail)
        setVerifiedNode_total: BigInt(0),
        setRole_total: BigInt(0),
        setNodeType_total: BigInt(0),
    };

    const UPDATER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("UPDATER_ROLE"));
    const MINTER_ROLE = ethers.keccak256(ethers.toUtf8Bytes("MINTER_ROLE")); // Added for minting

    // Setup runs once before all tests in this script
    before(async function () {
        signers = await ethers.getSigners();
        console.log(`Number of available signers: ${signers.length}`);
        // Need owner, mfg, retail, sn1, pn1, pn2, buyer = 7 signers for original full setup
        // Current usage seems to be owner, manufacturer, retailer, sn1, pn1, pn2, buyer (7)
        if (signers.length < 7) { 
            console.error("ERROR: Insufficient signers. This script requires at least 7 signers.");
            throw new Error(`Insufficient signers available. Need at least 7, but got ${signers.length}.`);
        }
        // Assign roles
        [owner, manufacturer, retailer, sn1, pn1, pn2, buyer] = signers.slice(0, 7);

        const t1 = pn1; // Transporter for releasePayment, reusing pn1

        console.log("\n--- Starting Initial Setup --- ");
        let tx, receipt;

        // 1. Deploy Contract (Gas cost NOT recorded as requested)
        console.log("Deploying SupplyChainNFT (Gas cost excluded from results)...");
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

        // 2. Grant Roles
        console.log("Granting Roles...");
        tx = await supplyChainNFT.connect(owner).grantRole(UPDATER_ROLE, sn1.address); // sn1 is history updater
        receipt = await tx.wait(1);
        results.grantRole_total += BigInt(receipt.gasUsed);
        console.log(`  Granted UPDATER_ROLE to ${sn1.address}. Gas: ${receipt.gasUsed}`);
        
        tx = await supplyChainNFT.connect(owner).grantRole(MINTER_ROLE, owner.address); // Grant MINTER_ROLE to owner for minting
        receipt = await tx.wait(1);
        results.grantRole_total += BigInt(receipt.gasUsed);
        console.log(`  Granted MINTER_ROLE to ${owner.address}. Gas: ${receipt.gasUsed}`);

        // 3. Register Nodes
        console.log("Registering Nodes...");
        const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 }; // Updated to match contract enum
        const ContractNodeType = { Primary: 0, Secondary: 1 };
        let regGasTotal = BigInt(0);

        async function registerNode(addr, isVerified, role, nodeType, name) {
            let nodeRegGas = BigInt(0);
            let r;
            console.log(`  Registering ${name} (${addr})...`);
            if (isVerified !== null) {
                tx = await supplyChainNFT.connect(owner).setVerifiedNode(addr, isVerified);
                r = await tx.wait(1);
                const gas = BigInt(r.gasUsed);
                nodeRegGas += gas;
                results.setVerifiedNode_total += gas;
                console.log(`    - setVerifiedNode: ${gas} gas`);
            }
            if (role !== null) {
                tx = await supplyChainNFT.connect(owner).setRole(addr, role);
                r = await tx.wait(1);
                const gas = BigInt(r.gasUsed);
                nodeRegGas += gas;
                results.setRole_total += gas;
                console.log(`    - setRole: ${gas} gas`);
            }
            if (nodeType !== null) {
                tx = await supplyChainNFT.connect(owner).setNodeType(addr, nodeType);
                r = await tx.wait(1);
                const gas = BigInt(r.gasUsed);
                nodeRegGas += gas;
                results.setNodeType_total += gas;
                console.log(`    - setNodeType: ${gas} gas`);
            }
            console.log(`  -> Total Gas for ${name}: ${nodeRegGas}`);
            return nodeRegGas;
        }

        regGasTotal += await registerNode(manufacturer.address, true, ContractRole.Manufacturer, null, "Manufacturer");
        regGasTotal += await registerNode(retailer.address, true, ContractRole.Retailer, null, "Retailer");
        regGasTotal += await registerNode(buyer.address, true, ContractRole.Customer, null, "Buyer");
        regGasTotal += await registerNode(sn1.address, true, null, ContractNodeType.Secondary, "Secondary Node (sn1)");
        regGasTotal += await registerNode(pn1.address, true, null, ContractNodeType.Primary, "Primary Node (pn1)");
        regGasTotal += await registerNode(pn2.address, true, null, ContractNodeType.Primary, "Primary Node (pn2)");
        regGasTotal += await registerNode(t1.address, null, ContractRole.Transporter, null, "Transporter (t1/pn1)");

        results.nodeRegistration_total = regGasTotal;
        console.log(`Total Node Registration Gas: ${results.nodeRegistration_total}`);
        console.log(`  Total setVerifiedNode Gas: ${results.setVerifiedNode_total}`);
        console.log(`  Total setRole Gas: ${results.setRole_total}`);
        console.log(`  Total setNodeType Gas: ${results.setNodeType_total}`);

        // 4. Mint Product NFT
        console.log("Minting NFT...");
        const mintParams = {
            recipient: manufacturer.address,
            uniqueProductID: "PROD-SETUP-TEST",
            batchNumber: "B-SETUP",
            manufacturingDate: "2025-05-01",
            expirationDate: "2026-05-01",
            productType: "TestProduct",
            manufacturerID: "MANU-TEST",
            quickAccessURL: "http://example.com/test",
            nftReference: "ipfs://nftSetupHash"
        };
        tx = await supplyChainNFT.connect(owner).mintNFT(mintParams); // Owner has MINTER_ROLE
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
        results.minting = BigInt(receipt.gasUsed);
        console.log(`NFT Minted with ID: ${tokenId}. Gas: ${results.minting}`);

        // 5. Store Initial CID (Simulated)
        initialPlaceholderCID = "bafkrei" + Math.random().toString(36).substring(2);
        console.log(`Storing initial dummy CID: ${initialPlaceholderCID}`);
        tx = await supplyChainNFT.connect(owner).storeInitialCID(tokenId, initialPlaceholderCID); // Owner has UPDATER_ROLE (or DEFAULT_ADMIN)
        receipt = await tx.wait(1);
        results.storeCID_initial = BigInt(receipt.gasUsed);
        console.log(`Initial CID stored. Gas: ${results.storeCID_initial}`);
        console.log("--- Initial Setup Complete ---");
    });

    it("Should measure gas for Marketplace Operations", async function () {
        console.log("\nStarting Marketplace Gas Measurement");
        const price = ethers.parseEther("0.01");
        let tx, receipt;
        const t1 = pn1; // Transporter for releasePayment

        expect(await supplyChainNFT.ownerOf(tokenId)).to.equal(manufacturer.address);

        // 1. List Product for Sale
        console.log(`   Listing product ${tokenId} for sale by ${manufacturer.address}`);
        tx = await supplyChainNFT.connect(manufacturer).sellProduct(tokenId, price);
        receipt = await tx.wait(1);
        results.sellProduct = BigInt(receipt.gasUsed);
        console.log(`     - sellProduct Gas: ${results.sellProduct}`);

        // 2. Initiate Purchase
        console.log(`   Initiating purchase for ${tokenId} by buyer ${buyer.address}`);
        tx = await supplyChainNFT.connect(buyer).initiatePurchase(tokenId, initialPlaceholderCID);
        receipt = await tx.wait(1);
        results.initiatePurchase = BigInt(receipt.gasUsed);
        console.log(`     - initiatePurchase Gas: ${results.initiatePurchase}`);

        // 3. Deposit Collateral
        console.log(`   Depositing collateral for ${tokenId} by buyer ${buyer.address}`);
        tx = await supplyChainNFT.connect(buyer).depositPurchaseCollateral(tokenId, { value: price });
        receipt = await tx.wait(1);
        results.depositCollateral = BigInt(receipt.gasUsed);
        console.log(`     - depositPurchaseCollateral Gas: ${results.depositCollateral}`);
        expect(await supplyChainNFT.ownerOf(tokenId)).to.equal(buyer.address);

        // 4. Release Payment
        console.log(`   Releasing payment for ${tokenId} by buyer ${buyer.address}`);
        tx = await supplyChainNFT.connect(buyer).releasePurchasePayment(tokenId, t1.address, true); // Assuming t1 is a valid transporter
        receipt = await tx.wait(1);
        results.releasePayment = BigInt(receipt.gasUsed);
        console.log(`     - releasePurchasePayment Gas: ${results.releasePayment}`);

        const totalMarketplaceGas = results.sellProduct + results.initiatePurchase + results.depositCollateral + results.releasePayment;
        console.log(`     - Total Gas for Marketplace Operations: ${totalMarketplaceGas}`);
    });

    after(async function () {
        if (results.minting === BigInt(0) && results.sellProduct === BigInt(0)) {
            console.log("\nSkipping final report as tests did not run.");
            return;
        }

        console.log("\n\n--- Gas Cost Analysis Results (Setup & Marketplace Only - Gas Units) ---");
        console.log("\n--- Setup Costs (One-time, Excludes Deployment) ---");
        console.log(`Grant Role (Total):         ${results.grantRole_total.toString().padStart(12)} Gas`);
        console.log(`Node Registration (Total):  ${results.nodeRegistration_total.toString().padStart(12)} Gas`);
        console.log(`  Total setVerifiedNode:  ${results.setVerifiedNode_total.toString().padStart(12)} Gas`);
        console.log(`  Total setRole:          ${results.setRole_total.toString().padStart(12)} Gas`);
        console.log(`  Total setNodeType:      ${results.setNodeType_total.toString().padStart(12)} Gas`);
        console.log(`Mint NFT:                   ${results.minting.toString().padStart(12)} Gas`);
        console.log(`Store Initial CID:          ${results.storeCID_initial.toString().padStart(12)} Gas`);
        const totalSetupGas = results.grantRole_total + results.nodeRegistration_total + results.minting + results.storeCID_initial;
        console.log(`-------------------------------------------------`);
        console.log(`TOTAL SETUP (Excl. Deploy): ${totalSetupGas.toString().padStart(12)} Gas`);

        console.log("\n--- Marketplace Costs (Per Sale Cycle) ---");
        console.log(`sellProduct:                ${results.sellProduct.toString().padStart(12)} Gas`);
        console.log(`initiatePurchase:           ${results.initiatePurchase.toString().padStart(12)} Gas`);
        console.log(`depositCollateral:          ${results.depositCollateral.toString().padStart(12)} Gas`);
        console.log(`releasePurchasePayment:     ${results.releasePayment.toString().padStart(12)} Gas`);
        const totalMarketplaceGas = results.sellProduct + results.initiatePurchase + results.depositCollateral + results.releasePayment;
        console.log(`-------------------------------------------------`);
        console.log(`TOTAL MARKETPLACE:          ${totalMarketplaceGas.toString().padStart(12)} Gas`);

        console.log("\n* Note: Deployment gas cost is excluded.");
        console.log("* Note: Ensure you have at least 7 signers available.");
    });
});

