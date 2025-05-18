// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

abstract contract DisputeResolution {
    mapping(uint256 => address[]) public arbitratorCandidates;
    mapping(uint256 => mapping(address => uint256)) public disputeVotes;
    mapping(uint256 => address) public selectedArbitrator;
    mapping(uint256 => bool) public disputeResolved;

    // --- Events ---
    event DisputeOpened(uint256 indexed tokenId);
    event ArbitratorVoted(uint256 indexed tokenId, address indexed voter, address indexed candidate);
    event ArbitratorSelected(uint256 indexed tokenId, address indexed arbitrator);
    event DisputeResolved(uint256 indexed tokenId, bool result);

    // Abstract function: must be implemented by the inheriting contract.
    // Function to get the product's history based on the tokenId
    function getProductHistory(uint256 tokenId) public view virtual returns (string memory);
    
    // Function to check if a candidate is verified
    function isVerified(address candidate) public view virtual returns (bool);

    // Open a dispute for a specific product
    function openDispute(uint256 tokenId, address[] memory candidates) public virtual {
        require(!disputeResolved[tokenId], "Dispute already resolved");
        
        // Ensure all candidates are verified before allowing them to be part of the arbitration
        for (uint256 i = 0; i < candidates.length; i++) {
            require(isVerified(candidates[i]), "Candidate not verified");
        }

        // Assign the candidates to the dispute
        arbitratorCandidates[tokenId] = candidates;
        emit DisputeOpened(tokenId);
    }

    // Vote for an arbitrator for a specific dispute
    function voteForArbitrator(uint256 tokenId, address candidate) public virtual {
        require(isCandidate(tokenId, candidate), "Not a valid arbitrator candidate");
        disputeVotes[tokenId][candidate] += 1;
        emit ArbitratorVoted(tokenId, msg.sender, candidate);
    }

    // Check if a candidate is part of the dispute's candidates list
    function isCandidate(uint256 tokenId, address candidate) internal view virtual returns (bool) {
        address[] memory candidates = arbitratorCandidates[tokenId];
        for (uint256 i = 0; i < candidates.length; i++) {
            if (candidates[i] == candidate) {
                return true;
            }
        }
        return false;
    }

    // Select the arbitrator with the highest votes
    function selectArbitrator(uint256 tokenId) public virtual {
        require(arbitratorCandidates[tokenId].length > 0, "No candidates available");
        
        address winner;
        uint256 highestVotes = 0;
        address[] memory candidates = arbitratorCandidates[tokenId];

        // Determine the candidate with the highest number of votes
        for (uint256 i = 0; i < candidates.length; i++) {
            address candidate = candidates[i];
            if (disputeVotes[tokenId][candidate] > highestVotes) {
                highestVotes = disputeVotes[tokenId][candidate];
                winner = candidate;
            }
        }
        
        require(winner != address(0), "No valid arbitrator selected");
        selectedArbitrator[tokenId] = winner;
        emit ArbitratorSelected(tokenId, winner);
    }

    // Resolve a dispute with a decision (true or false)
    function resolveDispute(uint256 tokenId, uint256 disputeBlockNumber, bool decision) public virtual {
        // Only the selected arbitrator can resolve the dispute
        require(msg.sender == selectedArbitrator[tokenId], "Only selected arbitrator can resolve");
        require(!disputeResolved[tokenId], "Dispute already resolved");

        // Get the product's history as evidence
        string memory evidence = getProductHistory(tokenId);
        require(bytes(evidence).length > 0, "No blockchain evidence found");

        // Ensure that the dispute resolution is happening within a valid block range
        require(block.number > disputeBlockNumber && block.number - disputeBlockNumber <= 256, "Block number out of range");

        // For demonstration purposes, the decision is directly used as the resolution result
        bool resolutionResult = decision;

        // Mark the dispute as resolved and emit the event
        disputeResolved[tokenId] = true;
        emit DisputeResolved(tokenId, resolutionResult);
    }
}
