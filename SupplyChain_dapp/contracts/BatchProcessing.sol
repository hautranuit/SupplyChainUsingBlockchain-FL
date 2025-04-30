// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "@openzeppelin/contracts/access/Ownable.sol";

abstract contract BatchProcessing is Ownable {
    // --- Data Structures ---
    struct TransactionData {
        address from;
        address to;
        uint256 tokenId;
    }

    struct Batch {
        uint256 batchId;
        TransactionData[] transactions;
        bool validated;
        bool committed;
        address proposer;
        bool flagged;
        address[] selectedValidators; // Store selected validators for this batch
        uint256 numSelectedValidators; // Store the count for easy access
    }

    mapping(uint256 => Batch) public batches;
    uint256 public nextBatchId = 1;
    mapping(uint256 => uint256) public batchApprovals;
    mapping(uint256 => uint256) public batchDenials;
    mapping(uint256 => mapping(address => uint8)) public batchVotes; // 0: not voted, 1: approve, 2: deny
    mapping(uint256 => mapping(address => bool)) public isSelectedValidator; // Tracks selected validators for a batch
    uint256 public numValidatorsToSelect = 5; // Configurable number of validators to select
    uint256 public superMajorityFraction = 66; // Percentage (e.g., 66 for 2/3)

    // --- Events ---
    event BatchProposed(uint256 batchId, address proposer, address[] selectedValidators);
    event BatchValidated(uint256 batchId, address validator, bool approve);
    event BatchCommitted(uint256 batchId, bool success);
    event ReconciliationComplete(uint256 reconciledCount);

    // --- Abstract Functions ---
    function _batchTransfer(address from, address to, uint256 tokenId) internal virtual;
    function ownerOf(uint256 tokenId) public view virtual returns (address);
    function updateReputation(address node, uint256 score) internal virtual;
    function penalizeNode(address node, uint256 penalty) internal virtual;
    function _getAllPrimaryNodes() internal view virtual returns (address[] memory);
    function getNodeReputation(address node) internal view virtual returns (uint256);
    function isPrimaryNode(address node) internal view virtual returns (bool);
    function isSecondaryNode(address node) internal view virtual returns (bool);
    // function getTotalPrimaryNodes() public view virtual returns (uint256); // No longer needed for commit logic

    // --- Batch Processing Functions ---

    // Propose a batch of transactions and select validators
    function proposeBatch(TransactionData[] memory txs) public virtual {
        require(isSecondaryNode(msg.sender), "Only Secondary Node can propose batch");

        uint256 currentBatchId = nextBatchId;
        Batch storage b = batches[currentBatchId];
        b.batchId = currentBatchId;
        b.proposer = msg.sender;

        for (uint256 i = 0; i < txs.length; i++) {
            // Optional: Add basic validation here if needed before adding to batch
            b.transactions.push(txs[i]);
        }

        // Select validators for this batch
        address[] memory selected = getRandomValidators(currentBatchId, numValidatorsToSelect);
        b.selectedValidators = selected;
        b.numSelectedValidators = selected.length; // Store the count

        // Update the mapping for quick lookup in validateBatch
        for (uint j = 0; j < selected.length; j++) {
            isSelectedValidator[currentBatchId][selected[j]] = true;
        }

        emit BatchProposed(currentBatchId, msg.sender, selected);
        nextBatchId++;
    }

    // Get a list of random validators based on reputation (weighted random selection)
    function getRandomValidators(uint256 batchId, uint256 numToSelect)
        internal view virtual returns (address[] memory) // Changed to internal
    {
        address[] memory primaryNodes = _getAllPrimaryNodes();
        uint256 len = primaryNodes.length;

        if (numToSelect == 0 || len == 0) {
            return new address[](0);
        }
        if (numToSelect > len) {
            numToSelect = len; // Select all if requested number exceeds available
        }

        uint256[] memory reputations = new uint256[](len);
        uint256 totalReputation = 0;
        for (uint256 i = 0; i < len; i++) {
            // Use a base reputation + actual reputation to avoid zero issues
            uint256 rep = 1 + getNodeReputation(primaryNodes[i]);
            reputations[i] = rep;
            totalReputation += rep;
        }

        address[] memory selected = new address[](numToSelect);
        address[] memory candidates = primaryNodes; // Use a copy to modify
        uint256[] memory currentReputations = reputations; // Use a copy

        // Use blockhash and batchId for pseudo-randomness, more secure than just timestamp
        uint256 randSeed = uint256(keccak256(abi.encodePacked(blockhash(block.number - 1), block.timestamp, batchId)));
        uint256 currentLength = len;

        for (uint256 i = 0; i < numToSelect; i++) {
            require(totalReputation > 0, "Reputation sum error"); // Should not happen if base rep is 1
            uint256 rand = randSeed % totalReputation;
            uint256 cumulative = 0;
            uint256 selectedIndex = 0; // Default to 0, will be overwritten

            // Find the selected candidate based on weighted probability
            for (uint256 j = 0; j < currentLength; j++) {
                cumulative += currentReputations[j];
                if (rand < cumulative) {
                    selectedIndex = j;
                    break;
                }
            }

            // Add selected candidate to the result
            selected[i] = candidates[selectedIndex];

            // Remove the selected candidate from the pool for next iteration (Fisher-Yates like step)
            totalReputation -= currentReputations[selectedIndex];
            candidates[selectedIndex] = candidates[currentLength - 1];
            currentReputations[selectedIndex] = currentReputations[currentLength - 1];
            currentLength--;

            // Update seed for next iteration
            randSeed = uint256(keccak256(abi.encodePacked(randSeed, i, selected[i])));
        }
        return selected;
    }

    // Validate the batch of transactions - only selected validators can vote
    function validateBatch(uint256 batchId, bool approve) public virtual {
        // require(isPrimaryNode(msg.sender), "Only Primary Node can validate"); // Redundant check?
        require(isSelectedValidator[batchId][msg.sender], "Not a selected validator for this batch");
        Batch storage b = batches[batchId];
        require(b.batchId != 0, "Batch does not exist"); // Check batch exists
        require(!b.committed, "Batch already committed");
        require(!b.flagged, "Batch flagged for review");
        require(batchVotes[batchId][msg.sender] == 0, "Already voted");

        batchVotes[batchId][msg.sender] = approve ? 1 : 2;

        if (approve) {
            batchApprovals[batchId]++;
        } else {
            batchDenials[batchId]++;
        }
        emit BatchValidated(batchId, msg.sender, approve);
    }

    // Commit the batch if the super majority of *selected* validators is met
    function commitBatch(uint256 batchId) public virtual {
        Batch storage b = batches[batchId];
        require(b.batchId != 0, "Batch does not exist");
        require(!b.committed, "Batch already committed");
        require(!b.flagged, "Batch is flagged, cannot commit");
        require(b.numSelectedValidators > 0, "No validators were selected for this batch"); // Sanity check

        // Calculate approval based on selected validators
        uint256 approvalPercent = (batchApprovals[batchId] * 100) / b.numSelectedValidators;

        if (approvalPercent >= superMajorityFraction) {
            b.validated = true;
            b.committed = true;
            // Execute transactions
            for (uint256 i = 0; i < b.transactions.length; i++) {
                TransactionData memory txData = b.transactions[i];
                // Ensure the 'from' address still owns the token before transferring
                if (ownerOf(txData.tokenId) == txData.from) {
                    _batchTransfer(txData.from, txData.to, txData.tokenId);
                } else {
                    // Optional: Handle cases where ownership changed or token doesn't exist
                    // E.g., emit an event, skip transfer
                }
            }
            // Reward proposer and validators
            updateReputation(b.proposer, 5); // Reward proposer for successful batch
            _rewardValidators(batchId, true);
            emit BatchCommitted(batchId, true);
        } else {
            // Check if denial threshold is met to flag (optional, could just fail commit)
            uint256 denialPercent = (batchDenials[batchId] * 100) / b.numSelectedValidators;
            // Example: Flag if more than 1/3 deny (100 - superMajorityFraction)
            if (denialPercent > (100 - superMajorityFraction)) {
                 b.flagged = true;
                 penalizeNode(b.proposer, 2); // Penalize proposer for failed batch
                 _rewardValidators(batchId, false); // Reward/penalize validators based on failed outcome
                 emit BatchCommitted(batchId, false); // Emit failure
            } else {
                // Not enough votes to commit or deny yet, do nothing, wait for more votes
            }
        }
    }

    // Reward/penalize validators based on their vote relative to the batch outcome
    function _rewardValidators(uint256 batchId, bool batchPassed) internal virtual {
        Batch storage b = batches[batchId];
        // Iterate only over selected validators for this batch
        for (uint256 i = 0; i < b.numSelectedValidators; i++) {
            address validator = b.selectedValidators[i];
            uint8 vote = batchVotes[batchId][validator];

            if (vote == 0) continue; // Skip if validator didn't vote

            bool votedApprove = (vote == 1);
            bool correctVote = (votedApprove == batchPassed);

            if (correctVote) {
                updateReputation(validator, 2); // Reward for correct vote
            } else {
                penalizeNode(validator, 1); // Penalize for incorrect vote
            }
        }
    }

    // Reconcile logs for flagged batches - Handles only flagged batches, not full periodic reconciliation
    // Note: Full periodic reconciliation across all nodes is complex on-chain.
    function reconcileLogs() public virtual onlyOwner {
        uint256 reconciledCount = 0;
        for (uint256 i = 1; i < nextBatchId; i++) {
            Batch storage b = batches[i];
            // Process batches that were flagged but never committed (e.g., due to insufficient denial votes)
            // Or re-evaluate previously flagged batches based on new criteria if needed.
            // Current logic: Simply unflags and penalizes proposer.
            if (b.flagged && !b.committed) {
                // Decision: What to do with flagged batches?
                // Option 1: Leave flagged, require manual intervention.
                // Option 2: Unflag after some time, penalize proposer (current implementation).
                // Option 3: Allow re-voting or re-proposal? (More complex)
                b.flagged = false; // Reset flag after review/action
                // Penalize proposer was already done when flagged in commitBatch
                // penalizeNode(b.proposer, 2);
                reconciledCount++;
            }
        }
        emit ReconciliationComplete(reconciledCount);
    }

    // --- Helper Functions ---
    function getBatchDetails(uint256 batchId) public view returns (Batch memory) {
         require(batches[batchId].batchId != 0, "Batch does not exist");
         return batches[batchId];
    }

     function getSelectedValidatorsForBatch(uint256 batchId) public view returns (address[] memory) {
         require(batches[batchId].batchId != 0, "Batch does not exist");
         return batches[batchId].selectedValidators;
     }
}

