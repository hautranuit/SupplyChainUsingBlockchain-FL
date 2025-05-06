// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./NFTCore.sol";

abstract contract Marketplace is NFTCore {
    mapping(uint256 => uint256) public listingPrice;

    // Removed: mapping(uint256 => mapping(uint256 => address)) public transportLegs;
    // Removed: mapping(uint256 => uint256[]) public distanceSegments;

    event ProductListedForSale(uint256 indexed tokenId, address indexed seller, uint256 price);
    event PurchaseInitiated(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price);
    // event PaymentReleaseTriggered(uint256 indexed tokenId, address indexed buyer); // Already in NFTCore or can be inferred
    event TransportStarted(uint256 indexed tokenId, address indexed owner, address[] transporters, string startLocation, string endLocation, uint256 distance);
    event TransportCompleted(uint256 indexed tokenId, address indexed completer);

    function sellProduct(uint256 tokenId, uint256 price) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(price > 0, "Marketplace: Price must be > 0");

        listingPrice[tokenId] = price;

        // Emit event for backend to handle IPFS logging
        emit ProductListedForSale(tokenId, msg.sender, price);
        // The old emit ProductHistoryUpdated(tokenId, history_string) is removed.
        // Backend will listen to ProductListedForSale, create IPFS log, and call updateProductHistory with new CID.
    }

    function initiatePurchase(uint256 tokenId, string memory CIDHash) public {
        uint256 price = listingPrice[tokenId];
        require(price > 0, "Marketplace: Product not listed for sale");

        address currentOwner = ownerOf(tokenId);

        string memory result = verifyProductAuthenticity(tokenId, CIDHash, currentOwner);
        require(keccak256(bytes(result)) == keccak256(bytes("Product is Authentic")), result);

        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.Listed || purchase.status == PurchaseStatus.Idle, "Marketplace: Purchase already in progress or not listed");

        purchase.buyer = msg.sender;
        purchase.seller = currentOwner;
        purchase.price = price;
        purchase.status = PurchaseStatus.AwaitingCollateral;

        listingPrice[tokenId] = 0; // Mark as no longer listed at this price

        emit PurchaseInitiated(tokenId, msg.sender, currentOwner, price);
    }

    function startTransport(
        uint256 tokenId,
        address[] memory transporters,
        string memory startLocation,
        string memory endLocation,
        uint256 distance
    ) public {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only owner can start transport");
        uint256 numTransporters = transporters.length;
        require(numTransporters > 0, "Marketplace: Must specify at least one transporter");

        // Removed on-chain storage of transportLegs and distanceSegments for gas optimization.
        // This data will be part of the event and logged to IPFS by the backend.
        // for (uint i = 0; i < numTransporters; i++) {
        //     transportLegs[tokenId][i] = transporters[i];
        // }
        // delete distanceSegments[tokenId]; 
        // for (uint i = 0; i < numSegments; i++) {
        //     distanceSegments[tokenId].push(i); 
        // }

        // Emit event for backend to handle IPFS logging
        emit TransportStarted(tokenId, msg.sender, transporters, startLocation, endLocation, distance);
        // The old emit ProductHistoryUpdated(tokenId, history_string) is removed.
        // Backend will listen to TransportStarted, create IPFS log, and call updateProductHistory with new CID.
    }

    function completeTransport(uint256 tokenId) public {
        // Require that the caller is one of the involved parties, e.g., the current owner or perhaps the last transporter.
        // For simplicity, keeping it as ownerOf for now, but this could be refined based on exact workflow.
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only NFT holder can complete transport");

        // Emit event for backend to handle IPFS logging
        emit TransportCompleted(tokenId, msg.sender);
        // The old emit ProductHistoryUpdated(tokenId, history_string) is removed.
        // Backend will listen to TransportCompleted, create IPFS log, and call updateProductHistory with new CID.
    }
}

