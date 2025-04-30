// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./Marketplace.sol";
import "./BatchProcessing.sol";
import "./DisputeResolution.sol";
import "./NodeManagement.sol";
import { IERC721 as IERC721Interface } from "@openzeppelin/contracts/token/ERC721/IERC721.sol";

contract SupplyChainNFT is Marketplace, BatchProcessing, DisputeResolution, NodeManagement {
    event PaymentSuspended(uint256 tokenId);
    event ReputationPenalized(address indexed node, uint256 newReputation);
    constructor(address initialOwner) NFTCore(initialOwner) {
        // NFTCore(initialOwner) sets the owner.
    }

    // --- Sale Functions (overriding Marketplace) ---
    function sellProduct(uint256 tokenId, uint256 price) public override {
        require(ownerOf(tokenId) == msg.sender, "You do not own this NFT");
        require(price > 0, "Price must be > 0");
        listingPrice[tokenId] = price;
        string memory productionHistory = string(abi.encodePacked("Products are ListedForSale "));
        emit ProductHistoryUpdated(tokenId, productionHistory);
    }
    
    function releasePaymentWithCollateral(
        uint256 tokenId,
        string memory CIDHash,
        address seller,
        address transporter,
        uint256 amount,
        bool meetsIncentiveCriteria
    ) public {
        string memory verificationStatus = verifyProductAuthenticity(tokenId, CIDHash, seller);
        require(keccak256(bytes(verificationStatus)) == keccak256(bytes("Product is Authentic")), verificationStatus);

        payable(seller).transfer(amount);
        if (meetsIncentiveCriteria) {
            uint256 incentive = amount / 10;
            payable(transporter).transfer(incentive);  
        }

        emit PaymentReleased(seller, amount);
    }
  
    function sellAndTransferProduct(
        uint256 tokenId,
        uint256 price,
        address buyer,
        string memory CIDHash
    ) public {
        // Verify authenticity before processing sale
        string memory result = verifyProductAuthenticity(
            tokenId,
            CIDHash,
            ownerOf(tokenId)
        );
        require(keccak256(bytes(result)) == keccak256(bytes("Product is Authentic")), result);
    
        // Transfer NFT ownership
        _transfer(msg.sender, buyer, tokenId);
        emit SaleSuccessful(tokenId, buyer, price);
        string memory productionHistory = string(abi.encodePacked("Products are saled successful on "));
        emit ProductHistoryUpdated(tokenId, productionHistory);
    }

    // --- Reputation Management ---
    // Internal function for contract logic
    function updateReputation(address node, uint256 score) internal override (BatchProcessing, NodeManagement) {
        nodeReputation[node] += score;
        emit ReputationUpdated(node, nodeReputation[node]);
    }

    // Public wrapper function for owner to manually update reputation
    function adminUpdateReputation(address node, uint256 score) public onlyOwner {
        updateReputation(node, score); // Calls the internal function
    }

    // Internal function for contract logic
    function penalizeNode(address node, uint256 penalty) internal override (BatchProcessing, NodeManagement) {
        if (nodeReputation[node] > penalty) {
            nodeReputation[node] -= penalty;
        } else {
            nodeReputation[node] = 0;
        }
        emit ReputationPenalized(node, nodeReputation[node]);
    }

    // Public wrapper function for owner to manually penalize node
    function adminPenalizeNode(address node, uint256 penalty) public onlyOwner {
        penalizeNode(node, penalty); // Calls the internal function
    }

    // --- Node Management Overrides ---
    function _getAllPrimaryNodes() internal view override returns (address[] memory) {
        uint256 totalPN = getTotalPrimaryNodes();
        address[] memory pns = new address[](totalPN);
        uint256 idx = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (_verifiedNodes[allNodes[i]] && nodeTypes[allNodes[i]] == NodeType.Primary) {
                pns[idx] = allNodes[i];
                idx++;
            }
        }
        return pns;
    }

    function getTotalPrimaryNodes() public view override(NodeManagement) returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (_verifiedNodes[allNodes[i]] && nodeTypes[allNodes[i]] == NodeType.Primary) {
                count++;
            }
        }
        return count;
    }

    function isPrimaryNode(address node) internal view override (BatchProcessing, NodeManagement) returns (bool) {
        return _verifiedNodes[node] && nodeTypes[node] == NodeType.Primary;
    }

    function isSecondaryNode(address node) internal view override (BatchProcessing, NodeManagement) returns (bool) {
        return _verifiedNodes[node] && nodeTypes[node] == NodeType.Secondary;
    }

    function getNodeReputation(address node) internal view override returns (uint256) {
        return nodeReputation[node];
    }

    // --- DisputeResolution Overrides ---
    function getProductHistory(uint256 tokenId) public view override returns (string memory) {
        return cidMapping[tokenId];
    }

    // --- Override ownerOf to resolve conflicts from multiple base classes ---
    function ownerOf(uint256 tokenId) public view override(ERC721, BatchProcessing, IERC721Interface) returns (address) {
        return ERC721.ownerOf(tokenId);
    }

    function _batchTransfer(address from, address to, uint256 tokenId) internal override {
    ERC721._transfer(from, to, tokenId);
    }

    function isVerified(address candidate) public view override (DisputeResolution, NodeManagement) returns (bool) {
    return _verifiedNodes[candidate];
    }
}
