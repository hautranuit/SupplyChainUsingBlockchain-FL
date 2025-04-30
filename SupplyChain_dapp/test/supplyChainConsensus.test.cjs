const { expect } = require("chai");
const { ethers } = require("hardhat");
const { performance } = require("perf_hooks"); // For timing

describe("SupplyChainNFT - Batch Processing Metrics", function () {
    let SupplyChainNFT, supplyChainNFT, owner, sn1, pn1, pn2, pn3, pn4, pn5, otherAccount;
    let signers;

    // Roles and Types Enum
    const Role = { Manufacturer: 0, Transporter: 1, Customer: 2, Retailer: 3 };
    const NodeType = { Primary: 0, Secondary: 1 };
    const superMajorityFraction = 66; // Match contract value

    // Batch Status Enum (based on contract logic)
    const BatchStatus = { Pending: 0, Committed: 1, Flagged: 2 }; // Added enum

    // Metrics storage
    let metrics = {
        proposalTimes: [],
        proposalGasCosts: [],
        validationTimes: [],
        validationGasCosts: [],
        commitTimesSuccess: [],
        commitGasCostsSuccess: [],
        commitTimesFail: [], // Track failed commit times separately if needed
        commitGasCostsFail: [], // Track failed commit gas separately if needed
        successfulBatches: 0,
        failedBatches: 0,
        totalBatches: 0,
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

    // Helper function to record metrics
    async function recordGasAndLog(txPromise, logLabel) {
        const tx = await txPromise;
        const receipt = await tx.wait();
        console.log(`Gas used for ${logLabel}: ${receipt.gasUsed.toString()}`);
        return receipt;
    }

    // Helper function to get initial reputations
    async function storeInitialReputations(nodes) {
        for (const node of nodes) {
            metrics.initialReps[node.address] = await supplyChainNFT.nodeReputation(node.address);
        }
    }

    // Helper function to calculate and record reputation changes
    async function recordReputationChanges(batchId, batchPassed, proposer, selectedValidators, validatorVotes) {
        // Proposer
        const finalProposerRep = await supplyChainNFT.nodeReputation(proposer.address);
        const initialProposerRep = BigInt(metrics.initialReps[proposer.address]); // Ensure BigInt
        const proposerRepChange = finalProposerRep - initialProposerRep; // Use BigInt subtraction
        if (batchPassed) {
            metrics.repChanges.correctProposer.push(proposerRepChange);
        } else {
            metrics.repChanges.incorrectProposer.push(proposerRepChange);
        }

        // Validators
        for (const validatorAddr of selectedValidators) {
            const finalRep = await supplyChainNFT.nodeReputation(validatorAddr);
            const initialRep = BigInt(metrics.initialReps[validatorAddr]); // Ensure BigInt
            const repChange = finalRep - initialRep; // Use BigInt subtraction
            const vote = validatorVotes[validatorAddr]; // vote: true (approve), false (deny)

            if (vote === undefined) continue; // Validator didn't vote (shouldn't happen in this test)

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
        // Get signers
        signers = await ethers.getSigners();
        [owner, sn1, pn1, pn2, pn3, pn4, pn5, otherAccount] = signers;

        // Deploy the contract
        SupplyChainNFT = await ethers.getContractFactory("SupplyChainNFT");
        supplyChainNFT = await SupplyChainNFT.deploy(owner.address);
        //await supplyChainNFT.deployed();

        // Configure Nodes
        const nodesToConfigure = [
            { signer: sn1, role: Role.Customer, type: NodeType.Secondary, rep: 10 },
            { signer: pn1, role: Role.Manufacturer, type: NodeType.Primary, rep: 50 },
            { signer: pn2, role: Role.Manufacturer, type: NodeType.Primary, rep: 40 },
            { signer: pn3, role: Role.Manufacturer, type: NodeType.Primary, rep: 60 },
            { signer: pn4, role: Role.Manufacturer, type: NodeType.Primary, rep: 55 },
            { signer: pn5, role: Role.Manufacturer, type: NodeType.Primary, rep: 45 },
        ];
        for (const node of nodesToConfigure) {
            await supplyChainNFT.connect(owner).setVerifiedNode(node.signer.address, true);
            await supplyChainNFT.connect(owner).setRole(node.signer.address, node.role);
            await supplyChainNFT.connect(owner).setNodeType(node.signer.address, node.type);
            if (node.rep > 0) {
                await supplyChainNFT.connect(owner).adminUpdateReputation(node.signer.address, node.rep);
            }
        }

        // Store initial reputations for all relevant nodes
        await storeInitialReputations([sn1, pn1, pn2, pn3, pn4, pn5]);

        // Mint Products using mintNFT (owned by SN1)
        const mintParams1 = { recipient: sn1.address, uniqueProductID: "PROD001", batchNumber: "B001", manufacturingDate: "D1", expirationDate: "E1", productType: "T1", manufacturerID: owner.address, quickAccessURL: "Q1", nftReference: "R1" };
        const mintTx1 = await supplyChainNFT.connect(owner).mintNFT(mintParams1);
        const receipt1 = await mintTx1.wait();

        // Lấy chữ ký hash của sự kiện ProductMinted trực tiếp (ethers v6)
        const productMintedSignature = "ProductMinted(address,uint256)";
        const productMintedTopic = ethers.id(productMintedSignature);
        // const productMintedTopic = supplyChainNFT.getEvent("ProductMinted").topicHash; // Dòng cũ
        console.log(`Expected ProductMinted Topic: ${productMintedTopic}`);

        // Tìm log thô dựa trên topic[0] và log các topic được kiểm tra
        let mintLog1 = null;
        console.log("Searching logs in receipt1:");
        for (const log of receipt1.logs) {
            console.log(`- Log Index ${log.index}, Topic 0: ${log.topics[0]}`);
            if (log.topics[0] === productMintedTopic) {
                mintLog1 = log;
                console.log(`  Found matching log at index ${log.index}`);
                break; // Thoát vòng lặp khi tìm thấy
            }
        }

        if (!mintLog1) {
            console.error("Receipt 1 Logs (không tìm thấy topic ProductMinted khớp):", JSON.stringify(receipt1.logs, null, 2));
            throw new Error("Không tìm thấy log thô khớp với topic ProductMinted trong biên nhận giao dịch mint đầu tiên.");
        }

        // Phân giải log tìm được (ethers v6)
        const mintEvent1 = supplyChainNFT.interface.parseLog({ topics: mintLog1.topics, data: mintLog1.data });
        const tokenId1 = mintEvent1.args.tokenId;
        console.log(`Minted tokenId1: ${tokenId1}`);

        // --- Làm tương tự cho lần mint thứ hai ---
        const mintParams2 = { recipient: sn1.address, uniqueProductID: "PROD002", batchNumber: "B002", manufacturingDate: "D2", expirationDate: "E2", productType: "T2", manufacturerID: owner.address, quickAccessURL: "Q2", nftReference: "R2" };
        const mintTx2 = await supplyChainNFT.connect(owner).mintNFT(mintParams2);
        const receipt2 = await mintTx2.wait();

        // Tìm log thô dựa trên topic[0] và log các topic được kiểm tra
        let mintLog2 = null;
        console.log("Searching logs in receipt2:");
        for (const log of receipt2.logs) {
            console.log(`- Log Index ${log.index}, Topic 0: ${log.topics[0]}`);
            if (log.topics[0] === productMintedTopic) {
                mintLog2 = log;
                console.log(`  Found matching log at index ${log.index}`);
                break;
            }
        }

        if (!mintLog2) {
            console.error("Receipt 2 Logs (không tìm thấy topic ProductMinted khớp):", JSON.stringify(receipt2.logs, null, 2));
            throw new Error("Không tìm thấy log thô khớp với topic ProductMinted trong biên nhận giao dịch mint thứ hai.");
        }

        const mintEvent2 = supplyChainNFT.interface.parseLog({ topics: mintLog2.topics, data: mintLog2.data });
        const tokenId2 = mintEvent2.args.tokenId;
        console.log(`Minted tokenId2: ${tokenId2}`);

        // Make tokenIds available in tests
        this.tokenId1 = tokenId1;
        this.tokenId2 = tokenId2;

    });

    // --- Test Cases --- 
    // Note: For more accurate averages, run these tests multiple times or loop within them.

    it("Scenario 1: Successful Batch Commit", async function () {
        const { tokenId1, tokenId2 } = this;
        metrics.totalBatches++; // Increment total batches counter

        // 1. Propose Batch (SN1 proposes transferring NFT 1 to PN1 and NFT 2 to PN2)
        const txData = [
            { from: sn1.address, to: pn1.address, tokenId: tokenId1 },
            { from: sn1.address, to: pn2.address, tokenId: tokenId2 }
        ];
        //console.log("txData to propose:", txData);

        const startTimePropose = performance.now();
        const proposeTx = await supplyChainNFT.connect(sn1).proposeBatch(txData);
        const receiptPropose = await proposeTx.wait();
        const endTimePropose = performance.now();
        metrics.proposalTimes.push(endTimePropose - startTimePropose);
        metrics.proposalGasCosts.push(receiptPropose.gasUsed);
        metrics.totalTransactions++;
        console.log(`Gas used for Propose: ${receiptPropose.gasUsed.toString()}`);

        // Tìm sự kiện BatchProposed bằng topic hash trực tiếp (ethers v6)
        const batchProposedSignature = "BatchProposed(uint256,address,address[])"; // <-- SỬA Ở ĐÂY
        const batchProposedTopic = ethers.id(batchProposedSignature);
        console.log(`Scenario 1 - Expected BatchProposed Topic: ${batchProposedTopic}`);
        let proposeLog = null;
        console.log("Scenario 1 - Searching propose logs:");
        for (const log of receiptPropose.logs) {
            console.log(`- Log Index ${log.index}, Topic 0: ${log.topics[0]}`);
            if (log.topics[0] === batchProposedTopic) {
                proposeLog = log;
                console.log(`  Found matching log at index ${log.index}`);
                break;
            }
        }

        if (!proposeLog) {
            console.error("Scenario 1 - Receipt Propose Logs (không tìm thấy topic BatchProposed):", JSON.stringify(receiptPropose.logs, null, 2));
            throw new Error("Scenario 1 - Không tìm thấy log thô khớp với topic BatchProposed.");
        }

        const proposeEvent = supplyChainNFT.interface.parseLog({ topics: proposeLog.topics, data: proposeLog.data });
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`Scenario 1 - Proposed Batch ID: ${batchId}`);
        console.log(`Scenario 1 - Selected Validators: ${selectedValidators}`);

        expect(selectedValidators.length).to.be.greaterThan(0);

        // 2. Validate Batch (Ensure Super-Majority Approve: >= 4/5 vote true)
        const validatorVotes = {}; // Track votes for reputation check
        let approvals = 0;
        const requiredApprovals = Math.ceil(selectedValidators.length * superMajorityFraction / 100); // Calculate required approvals (should be 4 for 5 validators and 66%)

        for (let i = 0; i < selectedValidators.length; i++) {
            const validatorAddr = selectedValidators[i];
            const validatorSigner = signers.find(s => s.address === validatorAddr);
            if (!validatorSigner) {
                console.warn(`Scenario 1: Could not find signer for selected validator ${validatorAddr}`);
                continue;
            }

            let vote = false; // Default to reject
            if (approvals < requiredApprovals) {
                vote = true; // Vote approve until supermajority is met
                approvals++;
            }

            const startTimeValidate = performance.now();
            const validateTx = await supplyChainNFT.connect(validatorSigner).validateBatch(batchId, vote);
            const receiptValidate = await validateTx.wait();
            const endTimeValidate = performance.now();
            metrics.validationTimes.push(endTimeValidate - startTimeValidate);
            metrics.validationGasCosts.push(receiptValidate.gasUsed);
            validatorVotes[validatorAddr] = vote;
        }
        console.log(`Scenario 1: Required ${requiredApprovals} approvals, got ${approvals} approvals from ${selectedValidators.length} validators.`);

        // 3. Commit Batch (Should succeed)
        const startTimeCommit = performance.now(); // Re-added missing line
        const commitTx = await supplyChainNFT.connect(sn1).commitBatch(batchId);
        const receiptCommit = await commitTx.wait();
        const endTimeCommit = performance.now();
        metrics.commitTimesSuccess.push(endTimeCommit - startTimeCommit);
        metrics.commitGasCostsSuccess.push(receiptCommit.gasUsed);
        metrics.successfulBatches++;
        metrics.totalTransactionsCommitted += txData.length; // Increment committed transactions
        await recordReputationChanges(batchId, true, sn1, selectedValidators, validatorVotes); // Record reputation changes

        // 4. Verify
        const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
        expect(batchDetails.committed).to.equal(true);
        expect(await supplyChainNFT.ownerOf(tokenId1)).to.equal(pn1.address);
        expect(await supplyChainNFT.ownerOf(tokenId2)).to.equal(pn2.address);
        // Add checks for reputation changes if needed
    });

    it("Scenario 2: Failed Batch Commit (Flagged)", async function () {
        const { tokenId1, tokenId2 } = this;
        metrics.totalBatches++; // Increment total batches counter

        // 1. Propose Batch (SN1 proposes transferring NFT 1 to PN1 and NFT 2 to PN2)
        const txData = [
            { from: sn1.address, to: pn1.address, tokenId: tokenId1 },
            { from: sn1.address, to: pn2.address, tokenId: tokenId2 }
        ];

        const startTimePropose = performance.now();
        const proposeTx = await supplyChainNFT.connect(sn1).proposeBatch(txData);
        const receiptPropose = await proposeTx.wait();
        const endTimePropose = performance.now();
        metrics.proposalTimes.push(endTimePropose - startTimePropose);
        metrics.proposalGasCosts.push(receiptPropose.gasUsed);
        metrics.totalTransactions++;
        console.log(`Gas used for Propose: ${receiptPropose.gasUsed.toString()}`);

        // Tìm sự kiện BatchProposed bằng topic hash trực tiếp (ethers v6)
        const batchProposedSignature = "BatchProposed(uint256,address,address[])"; // <-- SỬA Ở ĐÂY
        const batchProposedTopic = ethers.id(batchProposedSignature);
        console.log(`Scenario 2 - Expected BatchProposed Topic: ${batchProposedTopic}`);
        let proposeLog = null;
        console.log("Scenario 2 - Searching propose logs:");
        for (const log of receiptPropose.logs) {
            console.log(`- Log Index ${log.index}, Topic 0: ${log.topics[0]}`);
            if (log.topics[0] === batchProposedTopic) {
                proposeLog = log;
                console.log(`  Found matching log at index ${log.index}`);
                break;
            }
        }

        if (!proposeLog) {
            console.error("Scenario 2 - Receipt Propose Logs (không tìm thấy topic BatchProposed):", JSON.stringify(receiptPropose.logs, null, 2));
            throw new Error("Scenario 2 - Không tìm thấy log thô khớp với topic BatchProposed.");
        }

        const proposeEvent = supplyChainNFT.interface.parseLog({ topics: proposeLog.topics, data: proposeLog.data });
        const batchId = proposeEvent.args.batchId;
        const selectedValidators = proposeEvent.args.selectedValidators;
        console.log(`Scenario 2 - Proposed Batch ID: ${batchId}`);
        console.log(`Scenario 2 - Selected Validators: ${selectedValidators}`);

        expect(selectedValidators.length).to.be.greaterThan(0);

        // 2. Validate Batch (Majority Reject: PN1, PN3, PN4 vote false; PN2 votes true)
        const validatorVotes = {}; // Track votes for reputation check
        const validatorsToReject = [pn1, pn3, pn4];
        const validatorToApprove = pn2;

        for (const validator of validatorsToReject) {
            // Ensure validator was actually selected before voting
            if (selectedValidators.includes(validator.address)) {
                const startTimeValidate = performance.now();
                const validateTx = await supplyChainNFT.connect(validator).validateBatch(batchId, false);
                const receiptValidate = await validateTx.wait();
                const endTimeValidate = performance.now();
                metrics.validationTimes.push(endTimeValidate - startTimeValidate);
                metrics.validationGasCosts.push(receiptValidate.gasUsed);
                validatorVotes[validator.address] = false;
            } else {
                console.log(`Scenario 2: Validator ${validator.address} not selected, skipping reject vote.`);
            }
        }

        if (selectedValidators.includes(validatorToApprove.address)) {
            const startTimeValidateApprove = performance.now();
            const validateTxApprove = await supplyChainNFT.connect(validatorToApprove).validateBatch(batchId, true);
            const receiptValidateApprove = await validateTxApprove.wait();
            const endTimeValidateApprove = performance.now();
            metrics.validationTimes.push(endTimeValidateApprove - startTimeValidateApprove);
            metrics.validationGasCosts.push(receiptValidateApprove.gasUsed);
            validatorVotes[validatorToApprove.address] = true;
        } else {
            console.log(`Scenario 2: Validator ${validatorToApprove.address} not selected, skipping approve vote.`);
        }

        // 3. Commit Batch (Should fail and be flagged)
        const commitTx = await supplyChainNFT.connect(sn1).commitBatch(batchId);
        const receiptCommit = await commitTx.wait();
        metrics.failedBatches++; // Increment failed batches counter
        // metrics.totalTransactions++; // Count the failed commit attempt - Removed, not used for final metrics
        await recordReputationChanges(batchId, false, sn1, selectedValidators, validatorVotes); // Record reputation changes for failed batch

        // 4. Verify
        // Check for BatchCommitted event with success = false
        const batchCommittedSignature = "BatchCommitted(uint256,bool)";
        const batchCommittedTopic = ethers.id(batchCommittedSignature);
        let commitLog = null;
        for (const log of receiptCommit.logs) {
            if (log.topics[0] === batchCommittedTopic) {
                commitLog = log;
                break;
            }
        }
        expect(commitLog, "BatchCommitted event not found in commit receipt").to.not.be.null;
        const commitEvent = supplyChainNFT.interface.parseLog({ topics: commitLog.topics, data: commitLog.data });
        expect(commitEvent.args.success, "BatchCommitted event success should be false").to.be.false;
        expect(commitEvent.args.batchId, "BatchCommitted event batchId mismatch").to.equal(batchId);

        // Verify batch status is Flagged (hoặc Rejected)
        const batchDetails = await supplyChainNFT.getBatchDetails(batchId);
        expect(batchDetails.flagged, "Batch status should be Flagged").to.equal(true);

        // NFTs should still be owned by SN1
        expect(await supplyChainNFT.ownerOf(tokenId1)).to.equal(sn1.address);
        expect(await supplyChainNFT.ownerOf(tokenId2)).to.equal(sn1.address);
    });

    // --- Metrics Reporting --- 
    after(async function () {
        metrics.totalTestTimeEnd = performance.now();
        const totalDurationSeconds = (metrics.totalTestTimeEnd - metrics.totalTestTimeStart) / 1000;

        console.log("\n--- Performance & Cost Metrics ---");

        // Helper to calculate average BigNumber
        const avgBigNumber = (arr) => arr.length ? arr.reduce((acc, val) => acc + BigInt(val), BigInt(0)) / BigInt(arr.length) : BigInt(0);
        // Helper to calculate average number
        const avgNumber = (arr) => arr.length ? arr.reduce((acc, val) => acc + val, 0) / arr.length : 0;

        // Proposal Metrics
        const avgProposalTime = avgNumber(metrics.proposalTimes);
        const avgProposalGas = avgBigNumber(metrics.proposalGasCosts);
        console.log(`Average Proposal Time: ${avgProposalTime.toFixed(2)} ms`);
        console.log(`Average Proposal Gas Cost: ${avgProposalGas.toString()}`);
        console.log(`* Note: Selection time/gas is included within Proposal metrics as it's part of proposeBatch.`);

        // Validation Metrics
        const avgValidationTime = avgNumber(metrics.validationTimes);
        const avgValidationGas = avgBigNumber(metrics.validationGasCosts);
        console.log(`Average Validation Time (per vote): ${avgValidationTime.toFixed(2)} ms`);
        console.log(`Average Validation Gas Cost (per vote): ${avgValidationGas.toString()}`);

        // Commit Metrics
        const avgCommitTimeSuccess = avgNumber(metrics.commitTimesSuccess);
        const avgCommitGasSuccess = avgBigNumber(metrics.commitGasCostsSuccess);
        console.log(`Average Successful Commit Time: ${avgCommitTimeSuccess.toFixed(2)} ms`);
        console.log(`Average Successful Commit Gas Cost: ${avgCommitGasSuccess.toString()}`);
        // Add failed commit metrics if needed
        // const avgCommitTimeFail = avgNumber(metrics.commitTimesFail);
        // const avgCommitGasFail = avgBigNumber(metrics.commitGasCostsFail);
        // console.log(`Average Failed Commit Time: ${avgCommitTimeFail.toFixed(2)} ms`);
        // console.log(`Average Failed Commit Gas Cost: ${avgCommitGasFail.toString()}`);

        // Success Rate
        const successRate = metrics.totalBatches > 0 ? (metrics.successfulBatches / metrics.totalBatches) * 100 : 0;
        console.log(`Success Rate for Batches: ${successRate.toFixed(2)}% (${metrics.successfulBatches}/${metrics.totalBatches})`);

        // Reputation Changes
        const avgRepChangeCorrectVal = avgBigNumber(metrics.repChanges.correctValidator);
        const avgRepChangeIncorrectVal = avgBigNumber(metrics.repChanges.incorrectValidator);
        const avgRepChangeCorrectProp = avgBigNumber(metrics.repChanges.correctProposer);
        const avgRepChangeIncorrectProp = avgBigNumber(metrics.repChanges.incorrectProposer);
        console.log(`Average Reputation Change (Correct Validator): ${avgRepChangeCorrectVal.toString()}`);
        console.log(`Average Reputation Change (Incorrect Validator): ${avgRepChangeIncorrectVal.toString()}`);
        console.log(`Average Reputation Change (Correct Proposer): ${avgRepChangeCorrectProp.toString()}`);
        console.log(`Average Reputation Change (Incorrect Proposer): ${avgRepChangeIncorrectProp.toString()}`);

        // Throughput (TPS)
        const throughput = totalDurationSeconds > 0 ? metrics.totalTransactionsCommitted / totalDurationSeconds : 0;
        console.log(`Measured Throughput (Transactions Per Second): ${throughput.toFixed(2)} TPS`);
        console.log(`* Note: TPS measured in test environment (${metrics.totalTransactionsCommitted} txs / ${totalDurationSeconds.toFixed(2)} s). Actual TPS depends on network conditions.`);
        console.log(`* Note: Averages based on ${metrics.totalBatches} batch runs. Run more iterations for higher accuracy.`);
        console.log("----------------------------------\n");
    });
});

