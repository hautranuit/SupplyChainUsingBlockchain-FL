// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

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

    constructor(address initialOwner) Ownable(initialOwner) {
        // This constructor allows BatchProcessing to correctly initialize Ownable
    }

    // --- Abstract Functions ---
    function _batchTransfer(address from, address to, uint256 tokenId) internal virtual;
    function ownerOf(uint256 tokenId) public view virtual returns (address);
    function updateReputation(address node, int256 scoreChange, string memory reason) internal virtual;
    function penalizeNode(address node, uint256 penalty, string memory reason) internal virtual;
    function _getAllPrimaryNodes() internal view virtual returns (address[] memory);
    function getNodeReputation(address node) internal view virtual returns (uint256);
    function isPrimaryNode(address node) internal view virtual returns (bool);
    function isSecondaryNode(address node) internal view virtual returns (bool);

    // --- Batch Processing Functions ---

    function proposeBatch(TransactionData[] memory txs) public virtual {
        require(isSecondaryNode(msg.sender), "Only Secondary Node can propose batch");

        uint256 currentBatchId = nextBatchId;
        Batch storage b = batches[currentBatchId];
        b.batchId = currentBatchId;
        b.proposer = msg.sender;

        for (uint256 i = 0; i < txs.length; i++) {
            b.transactions.push(txs[i]);
        }

        address[] memory selected = getRandomValidators(currentBatchId, numValidatorsToSelect);
        b.selectedValidators = selected;
        b.numSelectedValidators = selected.length;

        for (uint j = 0; j < selected.length; j++) {
            isSelectedValidator[currentBatchId][selected[j]] = true;
        }

        emit BatchProposed(currentBatchId, msg.sender, selected);
        nextBatchId++;
    }

    function getRandomValidators(uint256 batchId, uint256 numToSelect)
        internal view virtual returns (address[] memory)
    {
        address[] memory primaryNodes = _getAllPrimaryNodes();
        uint256 len = primaryNodes.length;

        if (numToSelect == 0 || len == 0) {
            return new address[](0);
        }
        if (numToSelect > len) {
            numToSelect = len;
        }

        uint256[] memory reputations = new uint256[](len);
        uint256 totalReputation = 0;
        for (uint256 i = 0; i < len; i++) {
            uint256 rep = 1 + getNodeReputation(primaryNodes[i]);
            reputations[i] = rep;
            totalReputation += rep;
        }

        address[] memory selected = new address[](numToSelect);
        address[] memory candidates = primaryNodes;
        uint256[] memory currentReputations = reputations;

        uint256 randSeed = uint256(keccak256(abi.encodePacked(blockhash(block.number - 1), block.timestamp, batchId)));
        uint256 currentLength = len;

        for (uint256 i = 0; i < numToSelect; i++) {
            require(totalReputation > 0, "Reputation sum error");
            // Update random seed for each selection to ensure different entropy
            randSeed = uint256(keccak256(abi.encodePacked(randSeed, i, block.difficulty, block.coinbase)));
            uint256 rand = randSeed % totalReputation;
            uint256 cumulative = 0;
            uint256 selectedIndex = 0;

            for (uint256 j = 0; j < currentLength; j++) {
                cumulative += currentReputations[j];
                if (rand < cumulative) {
                    selectedIndex = j;
                    break;
                }
            }

            selected[i] = candidates[selectedIndex];
            totalReputation -= currentReputations[selectedIndex];
            candidates[selectedIndex] = candidates[currentLength - 1];
            currentReputations[selectedIndex] = currentReputations[currentLength - 1];
            currentLength--;
            randSeed = uint256(keccak256(abi.encodePacked(randSeed, i, selected[i])));
        }
        return selected;
    }

    function validateBatch(uint256 batchId, bool approve) public virtual {
        require(isSelectedValidator[batchId][msg.sender], "Not a selected validator for this batch");
        Batch storage b = batches[batchId];
        require(b.batchId != 0, "Batch does not exist");
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

    function commitBatch(uint256 batchId) public virtual {
        Batch storage b = batches[batchId];
        require(b.batchId != 0, "Batch does not exist");
        require(!b.committed, "Batch already committed");
        require(!b.flagged, "Batch is flagged, cannot commit");
        require(b.numSelectedValidators > 0, "No validators selected");

        uint256 approvalPercent = (batchApprovals[batchId] * 100) / b.numSelectedValidators;

        if (approvalPercent >= superMajorityFraction) {
            b.validated = true;
            b.committed = true;
            for (uint256 i = 0; i < b.transactions.length; i++) {
                TransactionData memory txData = b.transactions[i];
                if (ownerOf(txData.tokenId) == txData.from) {
                    _batchTransfer(txData.from, txData.to, txData.tokenId);
                }
            }
            updateReputation(b.proposer, 5, "Successful batch proposal");
            _rewardValidators(batchId, true);
            emit BatchCommitted(batchId, true);
        } else {
            uint256 denialPercent = (batchDenials[batchId] * 100) / b.numSelectedValidators;
            if (denialPercent > (100 - superMajorityFraction)) {
                 b.flagged = true;
                 penalizeNode(b.proposer, 2, "Batch flagged for review");
                 _rewardValidators(batchId, false);
                 emit BatchCommitted(batchId, false);
            }
        }
    }

    function _rewardValidators(uint256 batchId, bool batchPassed) internal virtual {
        Batch storage b = batches[batchId];
        for (uint256 i = 0; i < b.numSelectedValidators; i++) {
            address validator = b.selectedValidators[i];
            uint8 vote = batchVotes[batchId][validator];
            if (vote == 0) continue;
            bool votedApprove = (vote == 1);
            bool correctVote = (votedApprove == batchPassed);
            if (correctVote) {
                updateReputation(validator, 2, "Correct validation vote");
            } else {
                penalizeNode(validator, 1, "Incorrect validation vote");
            }
        }
    }

    function reconcileLogs() public virtual onlyOwner {
        uint256 reconciledCount = 0;
        for (uint256 i = 1; i < nextBatchId; i++) {
            Batch storage b = batches[i];
            if (b.flagged && !b.committed) {
                b.flagged = false;
                reconciledCount++;
            }
        }
        emit ReconciliationComplete(reconciledCount);
    }

    function getBatchDetails(uint256 batchId) public view returns (Batch memory) {
         require(batches[batchId].batchId != 0, "Batch does not exist");
         return batches[batchId];
    }

     function getSelectedValidatorsForBatch(uint256 batchId) public view returns (address[] memory) {
         require(batches[batchId].batchId != 0, "Batch does not exist");
         return batches[batchId].selectedValidators;
     }
}

