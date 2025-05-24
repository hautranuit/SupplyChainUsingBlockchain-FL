// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "./Marketplace.sol";
import "./BatchProcessing.sol";
import "./DisputeResolution.sol"; // Abstract contract
import "./NodeManagement.sol";
import { IERC721 as IERC721Interface } from "@openzeppelin/contracts/token/ERC721/IERC721.sol";
// import "@openzeppelin/contracts/utils/Context.sol"; // For _msgSender() if needed directly

contract SupplyChainNFT is Marketplace, BatchProcessing, DisputeResolution, NodeManagement {

    // --- Dispute Resolution State ---
    mapping(uint256 => DisputeResolution.Dispute) public disputesData; // disputeId => Dispute struct
    uint256 public nextDisputeId = 1;
    mapping(uint256 => uint256) public activeDisputeIdForToken; // tokenId => active disputeId (0 if none)

    // Add refundEnforced to the Dispute struct definition within SupplyChainNFT or ensure DisputeResolution.Dispute has it.
    // Assuming DisputeResolution.Dispute struct in the abstract contract is conceptual and the full struct is defined here implicitly by usage or explicitly if preferred.
    // For clarity, let's ensure our usage matches an expanded conceptual struct:
    /*
    struct Dispute {
        uint256 tokenId;
        address plaintiff; 
        address defendant; 
        string reason;
        string evidenceCID; 
        address[] candidates; 
        mapping(address => uint256) votes; 
        address selectedArbitrator;
        bool decisionRecorded;
        string resolutionDetails; 
        string resolutionCID;     
        uint8 outcome;            
        bool nftReturnEnforced; // Added for clarity if not implicitly part of 'enforced'
        bool refundEnforced;    // Added
        bool enforced; // General flag, can be true if all required actions are done
        uint256 openedTimestamp;
        uint256 decisionTimestamp;
        uint256 enforcedTimestamp;
    }
    */

    event PaymentSuspended(uint256 tokenId);
    // ReputationPenalized event is already in NodeManagement
    event DirectSaleAndTransferCompleted(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price, string oldCIDForVerification);

    constructor(address initialAdmin)
        NFTCore(initialAdmin) 
        BatchProcessing(initialAdmin) 
        // DisputeResolution itself is abstract and has no constructor
        // NodeManagement is abstract and has no constructor
    {
        // initialAdmin is owner for Ownable (from BatchProcessing) and has DEFAULT_ADMIN_ROLE (from NFTCore)
    }

    // --- Sale Functions (overriding Marketplace) ---
    function sellProduct(uint256 tokenId, uint256 price) public override {
        super.sellProduct(tokenId, price);
    }
    
    // --- Direct Sale ---
    function sellAndTransferProduct(
        uint256 tokenId,
        uint256 price,
        address buyer,
        string memory currentProductCID
    ) public {
        require(ownerOf(tokenId) == msg.sender, "SupplyChainNFT: Caller is not the owner");
        require(buyer != address(0), "SupplyChainNFT: Buyer cannot be zero address");

        string memory result = verifyProductAuthenticity(
            tokenId,
            currentProductCID,
            msg.sender 
        );
        require(keccak256(bytes(result)) == keccak256(bytes("Product is Authentic")), result);
    
        _transferNFTInternal(msg.sender, buyer, tokenId);
        
        emit DirectSaleAndTransferCompleted(tokenId, msg.sender, buyer, price, currentProductCID);
    }

    // --- Reputation Management Overrides ---
    function updateReputation(address node, int256 scoreChange, string memory reason) internal override(BatchProcessing, NodeManagement) {
        NodeManagement.updateReputation(node, scoreChange, reason);
    }

    function penalizeNode(address node, uint256 penalty, string memory reason) internal override(BatchProcessing, NodeManagement) {
        NodeManagement.penalizeNode(node, penalty, reason);
    }
    
    // --- Admin functions for reputation (using Ownable from BatchProcessing) ---
    function adminUpdateReputation(address node, uint256 score) public onlyOwner {
        updateReputation(node, int256(score), "Admin Reputation Update");
    }

    function adminPenalizeNode(address node, uint256 penalty) public onlyOwner {
        penalizeNode(node, penalty, "Admin Penalty");
    }

    // --- Node Management Overrides & Implementations ---
    // Concrete implementation for _getAllPrimaryNodes required by BatchProcessing
    function _getAllPrimaryNodes() internal view override(BatchProcessing) returns (address[] memory) {
        uint256 primaryNodeCount = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (isPrimaryNode(allNodes[i])) { // isPrimaryNode is from NodeManagement
                primaryNodeCount++;
            }
        }
        address[] memory primaryNodesList = new address[](primaryNodeCount);
        uint256 idx = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (isPrimaryNode(allNodes[i])) {
                primaryNodesList[idx] = allNodes[i];
                idx++;
            }
        }
        return primaryNodesList;
    }

    function getTotalPrimaryNodes() public view override(NodeManagement) returns (uint256) {
        // NodeManagement has an implementation, but we can also use the logic from _getAllPrimaryNodes if preferred
        // For consistency with the new _getAllPrimaryNodes, let's use its length.
        // Or, stick to NodeManagement's version if it's more optimized or has specific logic.
        // Sticking to NodeManagement's version as it was likely intended.
        return NodeManagement.getTotalPrimaryNodes(); 
    }

    function isPrimaryNode(address node) internal view override(BatchProcessing, NodeManagement) returns (bool) {
        return NodeManagement.isPrimaryNode(node);
    }

    function isSecondaryNode(address node) internal view override(BatchProcessing, NodeManagement) returns (bool) {
        return NodeManagement.isSecondaryNode(node);
    }

    function getNodeReputation(address node) internal view override(BatchProcessing) returns (uint256) {
        return NodeManagement.nodeReputation[node]; // Accessing public state variable from NodeManagement
    }

    // --- DisputeResolution Implementation ---

    /**
     * @dev Opens a new dispute for a given token.
     * @param tokenId The ID of the token being disputed.
     * @param reason A textual reason for the dispute.
     * @param evidenceDataString The data string supporting the dispute (to be processed by backend).
     * @return disputeId The ID of the newly created dispute.
     */
    function openDispute(
        uint256 tokenId,
        string memory reason,
        string memory evidenceDataString // MODIFIED: Was evidenceCID
    ) public returns (uint256) {
        require(_exists(tokenId), "Dispute: Token does not exist");
        require(activeDisputeIdForToken[tokenId] == 0, "Dispute: Token already has an active dispute");
        // require(ownerOf(tokenId) == msg.sender || /* other allowed parties like buyer from purchaseInfo */, "Dispute: Caller not authorized to open dispute");

        uint256 currentDisputeId = nextDisputeId++;
        DisputeResolution.Dispute storage newDispute = disputesData[currentDisputeId];
        
        newDispute.tokenId = tokenId;
        newDispute.plaintiff = msg.sender; 
        newDispute.reason = reason;
        // MODIFIED: Do not store data string directly into evidenceCID. Backend will update this.
        newDispute.evidenceCID = ""; // Or "pending_backend_upload"
        newDispute.openedTimestamp = block.timestamp;

        activeDisputeIdForToken[tokenId] = currentDisputeId;

        // MODIFIED: Emit evidenceDataString
        emit DisputeOpened(currentDisputeId, tokenId, msg.sender, reason, evidenceDataString, block.timestamp);
        return currentDisputeId;
    }

    function proposeArbitratorCandidate(uint256 disputeId, address candidate) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(!d.decisionRecorded, "Dispute: Decision already recorded, cannot add candidates");
        require(isVerified(candidate), "Dispute: Candidate must be a verified node"); 
        // Potentially add role check for arbitrator candidates if needed

        // Check if candidate already proposed
        for (uint i = 0; i < d.candidates.length; i++) {
            require(d.candidates[i] != candidate, "Dispute: Candidate already proposed");
        }
        d.candidates.push(candidate);
        emit ArbitratorCandidateProposed(disputeId, candidate, msg.sender, block.timestamp);
    }

    /**
     * @dev Records a decision for a dispute.
     * @param disputeId The ID of the dispute.
     * @param resolutionDetails Details of the resolution.
     * @param outcome The outcome of the dispute (e.g., 0 = unresolved, 1 = resolved in favor of plaintiff, 2 = resolved in favor of defendant).
     */
    function makeDisputeDecision(
        uint256 disputeId,
        string memory resolutionDetails,
        uint8 outcome
    ) public {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(!d.decisionRecorded, "Dispute: Decision already recorded");
        require(_isCandidate(disputeId, msg.sender), "Dispute: Caller is not an approved arbitrator");

        d.resolutionDetails = resolutionDetails;
        d.outcome = outcome;
        d.decisionRecorded = true;
        d.decisionTimestamp = block.timestamp;

        emit DisputeDecisionMade(disputeId, msg.sender, resolutionDetails, outcome, block.timestamp);
    }

    event DisputeDecisionMade(
        uint256 indexed disputeId,
        address indexed arbitrator,
        string resolutionDetails,
        uint8 outcome,
        uint256 timestamp
    );

    function _isCandidate(uint256 disputeId, address candidateAddr) internal view returns (bool) {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        if (d.openedTimestamp == 0) return false; // Dispute doesn't exist
        for (uint256 i = 0; i < d.candidates.length; i++) {
            if (d.candidates[i] == candidateAddr) {
                return true;
            }
        }
        return false;
    }

    function voteForArbitrator(uint256 disputeId, address candidate) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(_isCandidate(disputeId, candidate), "Dispute: Not a valid arbitrator candidate for this dispute");
        require(d.votes[msg.sender] == 0, "Dispute: Voter has already voted for a candidate in this dispute"); // Assuming one vote per address for simplicity
        // More complex voting rights can be added (e.g., only involved parties, or stake-based)
        
        d.votes[candidate] += 1; // Simple vote count
        d.votes[msg.sender] = 1; // Mark voter has voted (using candidate address as a placeholder for who they voted for, or a boolean flag)
                                 // A better way: mapping(uint256 => mapping(address => address)) public voterChoice;
                                 // For now, this simple check prevents double voting for *any* candidate by one address.

        emit ArbitratorVoted(disputeId, msg.sender, candidate, block.timestamp);
    }

    function selectArbitrator(uint256 disputeId) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(d.selectedArbitrator == address(0), "Dispute: Arbitrator already selected");
        require(d.candidates.length > 0, "Dispute: No candidates available");

        address winner = address(0);
        uint256 highestVotes = 0;

        for (uint256 i = 0; i < d.candidates.length; i++) {
            address currentCandidate = d.candidates[i];
            uint256 currentVotes = d.votes[currentCandidate]; // This assumes votes are stored per candidate
            if (currentVotes > highestVotes) {
                highestVotes = currentVotes;
                winner = currentCandidate;
            }
            // Tie-breaking logic could be added here if necessary (e.g., first proposed, random, etc.)
        }
        
        require(winner != address(0), "Dispute: No valid arbitrator selected (tie or no votes)");
        d.selectedArbitrator = winner;
        emit ArbitratorSelected(disputeId, d.tokenId, winner, block.timestamp);
    }

    function recordDecision(
        uint256 disputeId,
        string memory _resolutionDetails,
        string memory _resolutionDataString, // MODIFIED: Was _resolutionCID
        uint8 _outcome
    ) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(msg.sender == d.selectedArbitrator, "Dispute: Only selected arbitrator can record decision");
        require(!d.decisionRecorded, "Dispute: Decision already recorded");

        d.resolutionDetails = _resolutionDetails;
        // MODIFIED: Do not store data string directly into resolutionCID. Backend will update this.
        d.resolutionCID = ""; // Or "pending_backend_upload"
        d.outcome = _outcome;
        d.decisionRecorded = true;
        d.decisionTimestamp = block.timestamp;

        // MODIFIED: Emit _resolutionDataString
        emit DisputeDecisionRecorded(disputeId, d.tokenId, msg.sender, _resolutionDetails, _resolutionDataString, _outcome, block.timestamp);
    }

    function enforceNFTReturn(uint256 disputeId, address returnToAddress) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(msg.sender == d.selectedArbitrator, "Dispute: Only selected arbitrator can enforce");
        require(d.decisionRecorded, "Dispute: Decision must be recorded first");
        // require(!d.enforced, "Dispute: Actions already enforced or dispute concluded"); // More granular check below
        require(!d.nftReturnEnforced, "Dispute: NFT return already enforced");
        require(returnToAddress != address(0), "Dispute: Return address cannot be zero");

        uint256 currentTokenId = d.tokenId;
        address currentOwner = ownerOf(currentTokenId);
        
        require(currentOwner != returnToAddress, "Dispute: NFT already owned by the returnToAddress or invalid operation");

        _transferNFTInternal(currentOwner, returnToAddress, currentTokenId); 
        d.nftReturnEnforced = true;

        emit NFTReturnEnforced(disputeId, currentTokenId, msg.sender, currentOwner, returnToAddress, block.timestamp);
    }

    function enforceRefund(uint256 disputeId, address refundTo, address refundFrom, uint256 refundAmount) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(msg.sender == d.selectedArbitrator, "Dispute: Only selected arbitrator can enforce refund");
        require(d.decisionRecorded, "Dispute: Decision must be recorded first");
        require(!d.refundEnforced, "Dispute: Refund already enforced");
        require(refundTo != address(0), "Dispute: Refund address cannot be zero");
        require(refundAmount > 0, "Dispute: Refund amount must be positive");
        require(address(this).balance >= refundAmount, "Dispute: Contract has insufficient funds for this refund");

        d.refundEnforced = true; // Mark before transfer
        
        (bool success, ) = payable(refundTo).call{value: refundAmount}("");
        require(success, "Dispute: Refund transfer failed");

        emit RefundEnforced(disputeId, msg.sender, refundFrom, refundTo, refundAmount, block.timestamp);
    }

    function concludeDispute(uint256 disputeId) public override {
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        require(msg.sender == d.selectedArbitrator || msg.sender == d.plaintiff /* or contract owner for admin closure */, 
                "Dispute: Not authorized to conclude");
        require(d.decisionRecorded, "Dispute: Decision must be recorded to conclude");
        require(!d.enforced, "Dispute: Already concluded"); // Use general enforced flag for final conclusion
        
        // Determine if all necessary actions (based on outcome) were enforced.
        // This logic can be more sophisticated based on d.outcome.
        // For example, if outcome was FavorPlaintiff with NFT return and refund:
        // bool allActionsDone = d.nftReturnEnforced && d.refundEnforced;
        // If outcome was Dismissed, then no specific enforcement might be needed beyond recording decision.
        // For simplicity, we'll assume if it reaches here and decision is recorded, it can be marked enforced.
        // A more robust check would be based on the specific requirements of d.outcome.
        d.enforced = true; // General flag indicating the dispute process has run its course of enforcement actions.
        d.enforcedTimestamp = block.timestamp;
        activeDisputeIdForToken[d.tokenId] = 0; 

        emit DisputeConcluded(disputeId, d.tokenId, d.enforced, block.timestamp);
    }

    // --- ERC721 Overrides ---
    function ownerOf(uint256 tokenId) public view override(ERC721, BatchProcessing, IERC721Interface) returns (address) {
        return ERC721.ownerOf(tokenId);
    }

    // --- BatchProcessing Overrides ---
    function _batchTransfer(address from, address to, uint256 tokenId) internal override(BatchProcessing) {
        _transferNFTInternal(from, to, tokenId); // _transferNFTInternal is from NFTCore
    }

    // --- General Overrides ---
    function isVerified(address candidate) public view override(DisputeResolution, NodeManagement) returns (bool) {
        return NodeManagement.isVerified(candidate);
    }

    // --- NFTCore Overrides (if any specific to SupplyChainNFT context) ---
    // Example: If _transferNFTInternal needs special handling or events in SupplyChainNFT
    // function _transferNFTInternal(address from, address to, uint256 tokenId) internal override {
    //     super._transferNFTInternal(from, to, tokenId);
    //     // Additional logic specific to SupplyChainNFT context if needed
    // }


    // --- Required implementations for abstract functions from DisputeResolution.sol ---
    function getProductHistory(uint256 tokenId) public view override(DisputeResolution) returns (string memory) {
        // Assuming cidMapping in NFTCore stores the relevant history CID
        return cidMapping[tokenId]; 
    }

    // NEW: Implementation for updating CIDs by backend
    function updateDisputeEvidenceCID(uint256 disputeId, string memory newEvidenceCID) public override {
        require(msg.sender == owner(), "Caller is not the owner"); // Access control
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        d.evidenceCID = newEvidenceCID;
        // Optionally emit an event like:
        // emit DisputeEvidenceCIDUpdated(disputeId, newEvidenceCID);
    }

    function updateDisputeResolutionCID(uint256 disputeId, string memory newResolutionCID) public override {
        require(msg.sender == owner(), "Caller is not the owner"); // Access control
        DisputeResolution.Dispute storage d = disputesData[disputeId];
        require(d.openedTimestamp > 0, "Dispute: Dispute does not exist");
        // It makes sense that a decision must be recorded before its CID can be updated,
        // though the backend listener might call this after the DisputeDecisionRecorded event regardless.
        // require(d.decisionRecorded, "Dispute: Decision must be recorded first to update its CID");
        d.resolutionCID = newResolutionCID;
        // Optionally emit an event like:
        // emit DisputeResolutionCIDUpdated(disputeId, newResolutionCID);
    }

    // isVerified is already implemented/overridden from NodeManagement
    // function isVerified(address candidate) public view override(DisputeResolution, NodeManagement) returns (bool) {
    //     return super.isVerified(candidate);
    // }

    // Add a way for the contract to receive Ether
    receive() external payable {}
    fallback() external payable {}

    // Optional: A specific function for manufacturers to deposit funds for potential disputes
    function depositDisputeFunds() external payable {
        require(msg.value > 0, "Deposit amount must be greater than zero");
        // Optionally, log an event or associate funds with a specific manufacturer or product
        emit FundsDeposited(msg.sender, msg.value);
    }

    event FundsDeposited(address indexed depositor, uint256 amount); // New event
}

