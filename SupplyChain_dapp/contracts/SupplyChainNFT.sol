// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "./Marketplace.sol";
import "./BatchProcessing.sol";
import "./DisputeResolution.sol";
import "./NodeManagement.sol";
import { IERC721 as IERC721Interface } from "@openzeppelin/contracts/token/ERC721/IERC721.sol";

contract SupplyChainNFT is Marketplace, BatchProcessing, DisputeResolution, NodeManagement {

    event PaymentSuspended(uint256 tokenId);
    // ReputationPenalized event is already in NodeManagement
    event DirectSaleAndTransferCompleted(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price, string oldCIDForVerification);

    constructor(address initialAdmin)
        NFTCore(initialAdmin) 
        BatchProcessing(initialAdmin) 
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

    // --- DisputeResolution Overrides ---
    function getProductHistory(uint256 tokenId) public view override(DisputeResolution) returns (string memory) {
        // cidMapping is in NFTCore.sol, which Marketplace inherits. SupplyChainNFT inherits Marketplace.
        // So, cidMapping should be directly accessible.
        return cidMapping[tokenId]; 
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
}

