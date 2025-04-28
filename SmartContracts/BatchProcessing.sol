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
    }

    mapping(uint256 => Batch) public batches;
    uint256 public nextBatchId = 1;
    mapping(uint256 => uint256) public batchApprovals;
    mapping(uint256 => uint256) public batchDenials;
    mapping(uint256 => mapping(address => uint8)) public batchVotes;
    mapping(address => mapping(uint256 => bool)) public hasValidated;
    uint256 public superMajorityFraction = 66;

    // --- Events ---
    event BatchProposed(uint256 batchId, address proposer);
    event BatchValidated(uint256 batchId, address validator, bool approve);
    event BatchCommitted(uint256 batchId, bool success);

    // --- Abstract Functions ---
    function _batchTransfer(address from, address to, uint256 tokenId) internal virtual;
    function ownerOf(uint256 tokenId) public view virtual returns (address);
    function updateReputation(address node, uint256 score) internal virtual;
    function penalizeNode(address node, uint256 penalty) internal virtual;
    function _getAllPrimaryNodes() internal view virtual returns (address[] memory);
    function getNodeReputation(address node) internal view virtual returns (uint256);
    function isPrimaryNode(address node) internal view virtual returns (bool);
    function isSecondaryNode(address node) internal view virtual returns (bool);
    function getTotalPrimaryNodes() public view virtual returns (uint256);

    // --- Batch Processing Functions ---

    // Propose a batch of transactions
    function proposeBatch(TransactionData[] memory txs) public virtual {
        require(isSecondaryNode(msg.sender), "Only Secondary Node can propose batch");
        Batch storage b = batches[nextBatchId];
        b.batchId = nextBatchId;
        b.proposer = msg.sender;
        for (uint256 i = 0; i < txs.length; i++) {
            b.transactions.push(txs[i]);
        }
        emit BatchProposed(nextBatchId, msg.sender);
        nextBatchId++;
    }

    // Propose a batch of transactions with validation by Primary Nodes
    function proposeBatchWithTransactionValidation(TransactionData[] memory txs) public {
        require(isSecondaryNode(msg.sender), "Only Secondary Node can propose batch");
        // Group transactions and validate through Primary Nodes
        Batch storage newBatch = batches[nextBatchId];
        for (uint256 i = 0; i < txs.length; i++) {
            newBatch.transactions.push(txs[i]);
        }
        emit BatchProposed(nextBatchId, msg.sender);
        nextBatchId++;
    }

    // Get a list of random validators based on reputation
    function getRandomValidators(uint256 batchId, uint256 numValidators)
        public view virtual returns (address[] memory)
    {
        address[] memory primaryNodes = _getAllPrimaryNodes();
        uint256 len = primaryNodes.length;
        if (numValidators > len) {
            numValidators = len;
        }
        uint256[] memory reputations = new uint256[](len);
        uint256 totalReputation = 0;
        for (uint256 i = 0; i < len; i++) {
            uint256 rep = getNodeReputation(primaryNodes[i]);
            reputations[i] = rep;
            totalReputation += rep;
        }
        address[] memory selected = new address[](numValidators);
        uint256 randSeed = uint256(keccak256(abi.encodePacked(blockhash(uint64(block.number)), block.timestamp, batchId)));
        uint256 currentLength = len;
        for (uint256 i = 0; i < numValidators; i++) {
            require(totalReputation > 0, "No reputation points remaining");
            uint256 rand = randSeed % totalReputation;
            uint256 cumulative = 0;
            uint256 selectedIndex = 0;
            for (uint256 j = 0; j < currentLength; j++) {
                cumulative += reputations[j];
                if (rand < cumulative) {
                    selectedIndex = j;
                    break;
                }
            }
            selected[i] = primaryNodes[selectedIndex];
            totalReputation -= reputations[selectedIndex];
            primaryNodes[selectedIndex] = primaryNodes[currentLength - 1];
            reputations[selectedIndex] = reputations[currentLength - 1];
            currentLength--;
            randSeed = uint256(keccak256(abi.encodePacked(randSeed, i)));
        }
        return selected;
    }

    // Validate the batch of transactions
    function validateBatch(uint256 batchId, bool approve) public virtual {
        require(isPrimaryNode(msg.sender), "Only Primary Node can validate");
        Batch storage b = batches[batchId];
        require(!b.committed, "Batch already committed");
        require(!b.flagged, "Batch flagged for review");
        require(batchVotes[batchId][msg.sender] == 0, "Already voted");
        batchVotes[batchId][msg.sender] = approve ? 1 : 2;
        hasValidated[msg.sender][batchId] = true;
        if (approve) {
            batchApprovals[batchId]++;
        } else {
            batchDenials[batchId]++;
        }
        emit BatchValidated(batchId, msg.sender, approve);
    }

    // Commit the batch if the super majority is met
    function commitBatch(uint256 batchId) public virtual {
        Batch storage b = batches[batchId];
        require(!b.committed, "Batch already committed");
        require(!b.flagged, "Batch is flagged, cannot commit");
        uint256 totalPN = getTotalPrimaryNodes();
        uint256 approvalPercent = 0;
        if (totalPN > 0) {
            approvalPercent = (batchApprovals[batchId] * 100) / totalPN;
        }
        if (approvalPercent >= superMajorityFraction) {
            b.validated = true;
            b.committed = true;
            for (uint256 i = 0; i < b.transactions.length; i++) {
                TransactionData memory txData = b.transactions[i];
                if (ownerOf(txData.tokenId) == txData.from) {
                    _batchTransfer(txData.from, txData.to, txData.tokenId);
                }
            }
            updateReputation(b.proposer, 5);
            _rewardValidators(batchId, true);
            emit BatchCommitted(batchId, true);
        } else {
            b.flagged = true;
            emit BatchCommitted(batchId, false);
        }
    }

    // Reward validators based on their correct vote
    function _rewardValidators(uint256 batchId, bool batchPassed) internal virtual {
        bool correctVote = batchPassed;
        address[] memory pns = _getAllPrimaryNodes();
        for (uint256 i = 0; i < pns.length; i++) {
            address pn = pns[i];
            uint8 vote = batchVotes[batchId][pn];
            if (vote == 0) continue;
            bool votedApprove = (vote == 1);
            if (votedApprove == correctVote) {
                updateReputation(pn, 2);
            } else {
                penalizeNode(pn, 1);
            }
        }
    }

    // Reconcile logs for flagged batches
    function reconcileLogs() public virtual onlyOwner {
        for (uint256 i = 1; i < nextBatchId; i++) {
            Batch storage b = batches[i];
            if (b.flagged && !b.committed) {
                b.flagged = false;
                penalizeNode(b.proposer, 2);
            }
        }
    }
}
