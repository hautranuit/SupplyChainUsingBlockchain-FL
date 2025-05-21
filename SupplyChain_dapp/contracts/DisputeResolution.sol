// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

abstract contract DisputeResolution {
    // --- Data Structures ---
    struct Dispute {
        uint256 tokenId;
        address plaintiff; // Party who opened the dispute
        address defendant; // Counterparty (e.g., seller or previous owner)
        string reason;
        // MODIFIED: evidenceCID now stores the actual IPFS CID, updated by backend
        string evidenceCID; // CID for plaintiff's evidence (updated by backend)
        address[] candidates; // Arbitrator candidates
        mapping(address => uint256) votes; // Votes for each candidate
        address selectedArbitrator;
        bool decisionRecorded;
        string resolutionDetails; // Arbitrator's textual decision
        // MODIFIED: resolutionCID now stores the actual IPFS CID, updated by backend
        string resolutionCID;     // CID for arbitrator's resolution document (updated by backend)
        uint8 outcome;            // Enum-like: 0=Dismissed, 1=FavorPlaintiff, 2=FavorDefendant, 3=Partial
        bool nftReturnEnforced; // Added
        bool refundEnforced;    // Added
        bool enforced;
        uint256 openedTimestamp;
        uint256 decisionTimestamp;
        uint256 enforcedTimestamp;
    }

    // --- State Variables ---
    // Instead of direct mappings by tokenId, we'll use a disputeId.
    // The concrete contract (SupplyChainNFT) will manage the mapping from disputeId to Dispute struct.
    // It will also manage nextDisputeId and potentially a mapping from tokenId to activeDisputeId.

    // --- Events ---
    // Parameters updated to use disputeId and include more details
    // MODIFIED: DisputeOpened event now emits evidenceDataString (or placeholder) initially
    event DisputeOpened(uint256 indexed disputeId, uint256 indexed tokenId, address indexed plaintiff, string reason, string evidenceDataString, uint256 timestamp);
    event ArbitratorCandidateProposed(uint256 indexed disputeId, address indexed candidate, address indexed proposer, uint256 timestamp);
    event ArbitratorVoted(uint256 indexed disputeId, address indexed voter, address indexed candidate, uint256 timestamp);
    event ArbitratorSelected(uint256 indexed disputeId, uint256 indexed tokenId, address indexed selectedArbitrator, uint256 timestamp);
    
    // MODIFIED: DisputeDecisionRecorded event now emits resolutionDataString (or placeholder) initially
    event DisputeDecisionRecorded(
        uint256 indexed disputeId,
        uint256 indexed tokenId,
        address indexed arbitrator,
        string resolutionDetails,
        string resolutionDataString, // Was resolutionCID, now data string for backend to process
        uint8 outcome,
        uint256 timestamp
    );
    event NFTReturnEnforced(
        uint256 indexed disputeId,
        uint256 indexed tokenId,
        address indexed arbitrator,
        address from,
        address to,
        uint256 timestamp
    );
    event RefundEnforced(
        uint256 indexed disputeId,
        address arbitrator, // Removed indexed
        address indexed refundFrom, // Account deemed responsible for the refund
        address indexed refundTo,   // Account receiving the refund
        uint256 amount,
        uint256 timestamp
    );
    event DisputeConcluded(uint256 indexed disputeId, uint256 indexed tokenId, bool wasEnforced, uint256 timestamp);


    // --- Abstract Functions ---
    // These functions will be implemented in SupplyChainNFT.sol

    // Function to get the product's history based on the tokenId (remains the same)
    function getProductHistory(uint256 tokenId) public view virtual returns (string memory);
    
    // Function to check if a candidate is verified (remains the same)
    function isVerified(address candidate) public view virtual returns (bool);

    // --- Dispute Lifecycle Functions (Signatures for concrete implementation) ---

    // openDispute now needs to be defined in the concrete contract to return a disputeId
    // and handle the new Dispute struct.
    // The old signature: function openDispute(uint256 tokenId, address[] memory candidates) public virtual;
    // will be effectively replaced by a new one in SupplyChainNFT.sol that takes reason, evidenceCID etc.
    // NEW: Abstract functions for backend to update CIDs
    function updateDisputeEvidenceCID(uint256 disputeId, string memory newEvidenceCID) public virtual;
    function updateDisputeResolutionCID(uint256 disputeId, string memory newResolutionCID) public virtual;

    function proposeArbitratorCandidate(uint256 disputeId, address candidate) public virtual;

    function voteForArbitrator(uint256 disputeId, address candidate) public virtual;

    function selectArbitrator(uint256 disputeId) public virtual;

    function recordDecision(
        uint256 disputeId,
        string memory resolutionDetails,
        string memory resolutionDataString, // Was resolutionCID, now data string
        uint8 outcome
    ) public virtual;

    function enforceNFTReturn(uint256 disputeId, address returnToAddress) public virtual;

    function enforceRefund(uint256 disputeId, address refundTo, address refundFrom, uint256 amount) public virtual;

    // Future: function enforceRefund(...) public virtual;

    function concludeDispute(uint256 disputeId) public virtual;


    // Helper: Check if a candidate is part of the dispute's candidates list
    // This will be implemented in the concrete contract using the Dispute struct.
    // function isCandidate(uint256 disputeId, address candidate) internal view virtual returns (bool);

    // The old resolveDispute is now broken into recordDecision, enforce actions, and concludeDispute.
    // Old: function resolveDispute(uint256 tokenId, uint256 disputeBlockNumber, bool decision) public virtual;
}
