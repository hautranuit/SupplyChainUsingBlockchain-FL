const { expect } = require("chai");
const { ethers } = require("hardhat");
const { performance } = require("perf_hooks"); // For timing

describe("SupplyChainNFT - Batch Processing Metrics", function () {
    let supplyChainNFT, owner, sn1, pn1, pn2, pn3, pn4, pn5; // Removed otherAccount as it's not used
    let signers;

    // Roles and Types Enum - Align with your contract (NodeManagement.sol)
    const ContractRole = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3, Arbitrator: 4 }; 
    const ContractNodeType = { Primary: 0, Secondary: 1 };

    // Batch Status Enum (based on contract logic in BatchProcessing.sol)
    const BatchStatus = { Pending: 0, Committed: 1, Flagged: 2 };

    // Metrics storage
    let metrics = {
        proposalTimes: [],
        proposalGasCosts: [],
        validationTimes: [],
        validationGasCosts: [],
        commitTimesSuccess: [],
        commitGasCostsSuccess: [],
        // commitTimesFail: [], // Not explicitly tested for failure time in these scenarios
        // commitGasCostsFail: [], // Not explicitly tested for failure gas in these scenarios
        successfulBatches: 0,
        failedBatches: 0, // For Scenario 2
        totalBatches: 0,
        totalTransactionsInProposedBatches: 0, 
        totalTransactionsCommitted: 0,
        totalTestTimeStart: 0,
        totalTestTimeEnd: 0,
        repChanges: {
            correctValidator: [],
            incorrectValidator: [],
            correctProposer: [],
            incorrectProposer: []
        },
        initialReps: {}
    };

    // Helper function to store initial reputations
    async function storeInitialReputations(nodes) {
        for (const node of nodes) {
            metrics.initialReps[node.address] = await supplyChainNFT.nodeReputation(node.address);
        }
    }

    // Helper function to calculate and record reputation changes
    async function recordReputationChanges(batchId, batchPassed, proposer, selectedValidators, validatorVotes) {
        const finalProposerRep = await supplyChainNFT.nodeReputation(proposer.address);
        const initialProposerRep = BigInt(metrics.initialReps[proposer.address] || 0);
        const proposerRepChange = finalProposerRep - initialProposerRep;
        if (batchPassed) {
            metrics.repChanges.correctProposer.push(proposerRepChange);
        } else {
            metrics.repChanges.incorrectProposer.push(proposerRepChange);
        }

        for (const validatorAddr of selectedValidators) {
            const finalRep = await supplyChainNFT.nodeReputation(validatorAddr);
            const initialRep = BigInt(metrics.initialReps[validatorAddr] || 0);
            const repChange = finalRep - initialRep;
            const vote = validatorVotes[validatorAddr];

            if (vote === undefined) continue;

            const votedApprove = vote;
            const correctVote = (votedApprove === batchPassed);

            if (correctVote) {
                metrics.repChanges.correctValidator.push(repChange);
            } else {
                metrics.repChanges.incorrectValidator.push(repChange);
            }
        }
    }

    before(async function () {
        metrics.totalTestTimeStart = performance.now();
    });

    beforeEach(async function () {
        signers = await ethers.getSigners();
        if (signers.length < 7) {
            throw new Error(`Insufficient signers. Need at least 7, got ${signers.length}.`);
        }
        [owner, sn1, pn1, pn2, pn3, pn4, pn5] = signers.slice(0, 7);

        const SupplyChainNFTFactory = await ethers.getContractFactory("SupplyChainNFT");
        supplyChainNFT = await SupplyChainNFTFactory.deploy(owner.address);
        await supplyChainNFT.deploymentTransaction().wait(1);
        console.log(`SupplyChainNFT deployed at: ${await supplyChainNFT.getAddress()}`);

        const MINTER_ROLE = await supplyChainNFT.MINTER_ROLE();
        await supplyChainNFT.connect(owner).grantRole(MINTER_ROLE, owner.address);
        console.log(`Granted MINTER_ROLE to owner: ${owner.address}`);

        const nodesToConfigure = [
            { signer: sn1, role: ContractRole.Customer, type: ContractNodeType.Secondary, rep: 100 }, 
            { signer: pn1, role: ContractRole.Manufacturer, type: ContractNodeType.Primary, rep: 100 },
            { signer: pn2, role: ContractRole.Manufacturer, type: ContractNodeType.Primary, rep: 100 },
            { signer: pn3, role: ContractRole.Manufacturer, type: ContractNodeType.Primary, rep: 100 },
            { signer: pn4, role: ContractRole.Manufacturer, type: ContractNodeType.Primary, rep: 100 },
            { signer: pn5, role: ContractRole.Manufacturer, type: ContractNodeType.Primary, rep: 100 },
        ];
        console.log("Configuring nodes...");
        for (const node of nodesToConfigure) {
            await supplyChainNFT.connect(owner).setVerifiedNode(node.signer.address, true);
            await supplyChainNFT.connect(owner).setRole(node.signer.address, node.role);
            await supplyChainNFT.connect(owner).setNodeType(node.signer.address, node.type);
            if (node.rep > 0) {
                await supplyChainNFT.connect(owner).adminUpdateReputation(node.signer.address, node.rep);
            }
            console.log(`  Configured node: ${node.signer.address} with Role: ${node.role}, Type: ${node.type}, Rep: ${node.rep}`);
        }
        await storeInitialReputations([sn1, pn1, pn2, pn3, pn4, pn5]);
        console.log("Initial reputations stored.");

        console.log("Minting NFTs...");
        const mintParams1 = { recipient: sn1.address, uniqueProductID: "PROD001", batchNumber: "B001", manufacturingDate: "D1", expirationDate: "E1", productType: "T1", manufacturerID: "M1", quickAccessURL: "Q1", nftReference: "R1" };
        let tx = await supplyChainNFT.connect(owner).mintNFT(mintParams1);
        let receipt = await tx.wait(1);
        let mintEvent = receipt.logs.map(log => { try { return supplyChainNFT.interface.parseLog(log); } catch (e) { return null; } }).find(e => e && e.name === "ProductMinted");
        if (!mintEvent) throw new Error("ProductMinted event not found for tokenId1.");
        this.tokenId1 = mintEvent.args.tokenId;
        console.log(`  Minted tokenId1: ${this.tokenId1}`);

        const mintParams2 = { recipient: sn1.address, uniqueProductID: "PROD002", batchNumber: "B002", manufacturingDate: "D2", expirationDate: "E2", productType: "T2", manufacturerID: "M2", quickAccessURL: "Q2", nftReference: "R2" };
        tx = await supplyChainNFT.connect(owner).mintNFT(mintParams2);
        receipt = await tx.wait(1);
        mintEvent = receipt.logs.map(log => { try { return supplyChainNFT.interface.parseLog(log); } catch (e) { return null; } }).find(e => e && e.name === "ProductMinted");
        if (!mintEvent) throw new Error("ProductMinted event not found for tokenId2.");
        this.tokenId2 = mintEvent.args.tokenId;
        console.log(`  Minted tokenId2: ${this.tokenId2}`);
        console.log("NFTs minted and setup complete.");
    });

    it("Scenario 1: Successful Batch Commit", async function () {
        console.log("\nStarting Scenario 1: Successful Batch Commit");
        const { tokenId1, tokenId2 } = this;
        metrics.totalBatches++;

        const txData = [
            { from: sn1.address, to: pn1.address, tokenId: tokenId1 },
            { from: sn1.address, to: pn2.address, tokenId: tokenId2 }
        ];
        metrics.totalTransactionsInProposedBatches += txData.length;
        console.log(`  Proposing batch with ${txData.length} transactions...`);

        const startTimePropose = performance.now();
        const proposeTx = await supplyChainNFT.connect(sn1).proposeBatch(txData);
        const receiptPropose = await proposeTx.wait();
        const endTimePropose = performance.now();
        metrics.proposalTimes.push(endTimePropose - startTimePropose);
        metrics.proposalGasCosts.push(receiptPropose.gasUsed);
        console.log(`    Propose Batch Gas: ${receiptPropose.gasUsed}`);

        const proposeEvent = receiptPropose.logs.map(log => { try { return supplyChainNFT.interface.parseLog(log); } catch (e) { return null; } }).find(e => e && e.name === "BatchProposed");
        if (!proposeEvent) throw new Error("BatchProposed event not found.");
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`    Batch ID: ${batchId}, Selected Validators: ${selectedValidators.length}`);
        expect(selectedValidators.length).to.be.greaterThan(0);

        console.log("  Validators voting...");
        const validatorVotes = {};
        let approvals = 0;
        const superMajority = await supplyChainNFT.superMajorityFraction();
        const requiredApprovals = Math.ceil(selectedValidators.length * Number(superMajority) / 100);

        for (let i = 0; i < selectedValidators.length; i++) {
            const validatorAddr = selectedValidators[i];
            const validatorSigner = signers.find(s => s.address === validatorAddr);
            if (!validatorSigner) {
                console.warn(`    Could not find signer for validator ${validatorAddr}, skipping vote.`);
                continue;
            }
            let vote = approvals < requiredApprovals; 
            if (vote) approvals++;

            const startTimeValidate = performance.now();
            const validateTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote);
            const receiptValidate = await validateTx.wait();
            const endTimeValidate = performance.now();
            metrics.validationTimes.push(endTimeValidate - startTimeValidate);
            metrics.validationGasCosts.push(receiptValidate.gasUsed);
            validatorVotes[validatorAddr] = vote;
            console.log(`    Validator ${validatorSigner.address} voted: ${vote}. Gas: ${receiptValidate.gasUsed}`);
        }
        console.log(`    Total approvals: ${approvals} out of ${selectedValidators.length} (required: ${requiredApprovals})`);

        console.log("  Committing batch...");
        const startTimeCommit = performance.now();
        const commitTx = await supplyChainNFT.connect(sn1).commitBatch(batchId);
        const receiptCommit = await commitTx.wait();
        const endTimeCommit = performance.now();
        metrics.commitTimesSuccess.push(endTimeCommit - startTimeCommit);
        metrics.commitGasCostsSuccess.push(receiptCommit.gasUsed);
        metrics.successfulBatches++;
        metrics.totalTransactionsCommitted += txData.length;
        await recordReputationChanges(batchId, true, sn1, selectedValidators, validatorVotes);
        console.log(`    Commit Batch Gas: ${receiptCommit.gasUsed}`);

        const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
        // Original: expect(batchDetails.status).to.equal(BatchStatus.Committed);
        // Direct checks are more robust here:
        expect(batchDetails.committed).to.equal(true, "Batch should be committed");
        expect(batchDetails.flagged).to.equal(false, "Batch should not be flagged for successful commit");
        expect(await supplyChainNFT.ownerOf(tokenId1)).to.equal(pn1.address);
        expect(await supplyChainNFT.ownerOf(tokenId2)).to.equal(pn2.address);
        console.log("  Scenario 1: Batch committed successfully and ownership verified.");
    });

    it("Scenario 2: Failed Batch Commit (Flagged due to insufficient approvals)", async function () {
        console.log("\nStarting Scenario 2: Failed Batch Commit (Flagged)");
        const { tokenId1 } = this;
        metrics.totalBatches++;

        const txData = [{ from: sn1.address, to: pn3.address, tokenId: tokenId1 }];
        metrics.totalTransactionsInProposedBatches += txData.length;
        console.log(`  Proposing batch with ${txData.length} transaction...`);

        const proposeTx = await supplyChainNFT.connect(sn1).proposeBatch(txData);
        const receiptPropose = await proposeTx.wait();
        metrics.proposalGasCosts.push(receiptPropose.gasUsed);
        console.log(`    Propose Batch Gas: ${receiptPropose.gasUsed}`);

        const proposeEvent = receiptPropose.logs.map(log => { try { return supplyChainNFT.interface.parseLog(log); } catch (e) { return null; } }).find(e => e && e.name === "BatchProposed");
        if (!proposeEvent) throw new Error("BatchProposed event not found.");
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`    Batch ID: ${batchId}, Selected Validators: ${selectedValidators.length}`);

        console.log("  Validators voting (all false)...");
        const validatorVotes = {};
        for (const validatorAddr of selectedValidators) {
            const validatorSigner = signers.find(s => s.address === validatorAddr);
            if (!validatorSigner) {
                console.warn(`    Could not find signer for validator ${validatorAddr}, skipping vote.`);
                continue;
            }
            const validateTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, false);
            const receiptValidate = await validateTx.wait();
            metrics.validationGasCosts.push(receiptValidate.gasUsed);
            validatorVotes[validatorAddr] = false;
            console.log(`    Validator ${validatorSigner.address} voted: false. Gas: ${receiptValidate.gasUsed}`);
        }

        console.log("  Attempting to commit batch (should be flagged)...");
        const commitTx = await supplyChainNFT.connect(sn1).commitBatch(batchId);
        const receiptCommit = await commitTx.wait(); 
        metrics.commitGasCostsSuccess.push(receiptCommit.gasUsed); // Still record gas, could be commitGasCostsFail if distinct
        metrics.failedBatches++;
        await recordReputationChanges(batchId, false, sn1, selectedValidators, validatorVotes);
        console.log(`    Commit Attempt Gas (Flagged): ${receiptCommit.gasUsed}`);

        const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
        // Original: expect(batchDetails.status).to.equal(BatchStatus.Flagged);
        // Direct checks are more robust here:
        expect(batchDetails.committed).to.equal(false, "Batch should not be committed if flagged");
        expect(batchDetails.flagged).to.equal(true, "Batch should be flagged");
        expect(await supplyChainNFT.ownerOf(tokenId1)).to.equal(sn1.address); // Ownership should not change
        console.log("  Scenario 2: Batch flagged successfully and ownership verified as unchanged.");
    });

    after(async function () {
        metrics.totalTestTimeEnd = performance.now();
        const totalDurationSeconds = (metrics.totalTestTimeEnd - metrics.totalTestTimeStart) / 1000;

        console.log("\n\n--- Batch Processing Metrics Summary ---");
        console.log(`Total Test Duration: ${totalDurationSeconds.toFixed(2)} seconds`);
        console.log(`Total Batches Processed: ${metrics.totalBatches}`);
        console.log(`  Successful Batches: ${metrics.successfulBatches}`);
        console.log(`  Failed/Flagged Batches: ${metrics.failedBatches}`);
        console.log(`Total Individual Transactions in Proposed Batches: ${metrics.totalTransactionsInProposedBatches}`);
        console.log(`Total Individual Transactions Committed: ${metrics.totalTransactionsCommitted}`);

        const avgProposalGas = metrics.proposalGasCosts.length > 0 ? metrics.proposalGasCosts.reduce((a, b) => a + b, BigInt(0)) / BigInt(metrics.proposalGasCosts.length) : BigInt(0);
        const avgValidationGasPerVote = metrics.validationGasCosts.length > 0 ? metrics.validationGasCosts.reduce((a, b) => a + b, BigInt(0)) / BigInt(metrics.validationGasCosts.length) : BigInt(0);
        const avgCommitGasSuccess = metrics.commitGasCostsSuccess.length > 0 ? metrics.commitGasCostsSuccess.reduce((a, b) => a + b, BigInt(0)) / BigInt(metrics.commitGasCostsSuccess.length) : BigInt(0);

        console.log("\nAverage Gas Costs:");
        console.log(`  Propose Batch: ${avgProposalGas.toString()} Gas`);
        console.log(`  Validate Batch (per validator vote): ${avgValidationGasPerVote.toString()} Gas`);
        console.log(`  Commit Batch (Successful/Flagged): ${avgCommitGasSuccess.toString()} Gas`);
        
        console.log("\nReputation Changes (Delta):");
        console.log(`  Correct Proposer: ${metrics.repChanges.correctProposer.map(r => r.toString()).join(', ') || 'N/A'}`);
        console.log(`  Incorrect Proposer: ${metrics.repChanges.incorrectProposer.map(r => r.toString()).join(', ') || 'N/A'}`);
        console.log(`  Correct Validator: ${metrics.repChanges.correctValidator.map(r => r.toString()).join(', ') || 'N/A'}`);
        console.log(`  Incorrect Validator: ${metrics.repChanges.incorrectValidator.map(r => r.toString()).join(', ') || 'N/A'}`);
        
        console.log("\n* Notes: Gas costs are per transaction. Validation gas is per individual vote.");
        console.log("* Ensure contract enums for roles, types, and batch status match this script.");
    });
});

