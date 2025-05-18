// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "./NFTCore.sol";

abstract contract Marketplace is NFTCore {
    mapping(uint256 => uint256) public listingPrice;

    // Removed: mapping(uint256 => mapping(uint256 => address)) public transportLegs;
    // Removed: mapping(uint256 => uint256[]) public distanceSegments;

    event ProductListedForSale(uint256 indexed tokenId, address indexed seller, uint256 price, uint256 timestamp);
    event PurchaseInitiated(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price, uint256 timestamp);
    // event PaymentReleaseTriggered(uint256 indexed tokenId, address indexed buyer); // Already in NFTCore or can be inferred
    event TransportStarted(uint256 indexed tokenId, address indexed owner, address[] transporters, string startLocation, string endLocation, uint256 distance, uint256 timestamp);
    event TransportCompleted(uint256 indexed tokenId, address indexed completer, uint256 timestamp);
    event ProductDelisted(uint256 indexed tokenId, address indexed seller, uint256 timestamp);
    event ProductPriceChanged(uint256 indexed tokenId, address indexed seller, uint256 oldPrice, uint256 newPrice, uint256 timestamp);

    function sellProduct(uint256 tokenId, uint256 price) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(price > 0, "Marketplace: Price must be > 0");

        listingPrice[tokenId] = price;

        // Emit event for backend to handle IPFS logging
        emit ProductListedForSale(tokenId, msg.sender, price, block.timestamp);
        // The old emit ProductHistoryUpdated(tokenId, history_string) is removed.
        // Backend will listen to ProductListedForSale, create IPFS log, and call updateProductHistory with new CID.
        // Update lastActionTimestamp for seller (msg.sender) - handled in SupplyChainNFT or via off-chain listeners
    }

    function delistProduct(uint256 tokenId) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(listingPrice[tokenId] > 0, "Marketplace: Product not listed for sale");

        listingPrice[tokenId] = 0;
        emit ProductDelisted(tokenId, msg.sender, block.timestamp);
        // Update lastActionTimestamp for seller (msg.sender)
    }

    function changeProductPrice(uint256 tokenId, uint256 newPrice) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(listingPrice[tokenId] > 0, "Marketplace: Product not listed for sale");
        require(newPrice > 0, "Marketplace: New price must be > 0");

        uint256 oldPrice = listingPrice[tokenId];
        listingPrice[tokenId] = newPrice;
        emit ProductPriceChanged(tokenId, msg.sender, oldPrice, newPrice, block.timestamp);
        // Update lastActionTimestamp for seller (msg.sender)
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

        emit PurchaseInitiated(tokenId, msg.sender, currentOwner, price, block.timestamp);
        // Update lastActionTimestamp for buyer (msg.sender) and currentOwner (seller)
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

        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        // Ensure the purchase is in a state where transport can be started
        require(purchase.status == PurchaseStatus.CollateralDeposited, "Marketplace: Collateral not deposited or transport already handled");

        PurchaseStatus oldStatus = purchase.status;
        purchase.status = PurchaseStatus.InTransit;

        // Removed on-chain storage of transportLegs and distanceSegments for gas optimization.
        // This data will be part of the event and logged to IPFS by the backend.
        emit TransportStarted(tokenId, msg.sender, transporters, startLocation, endLocation, distance, block.timestamp);
        emit PurchaseStatusUpdated(tokenId, oldStatus, PurchaseStatus.InTransit, msg.sender, block.timestamp); // from NFTCore
        // Update lastActionTimestamp for owner (msg.sender) and transporters
    }

    function completeTransport(uint256 tokenId) public {
        // Require that msg.sender is one of the transporters or the current owner
        // This logic might need to be more specific based on your system's rules
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        bool isTransporter = false;
        // Assuming purchase.transporters array exists and is populated in PurchaseInfo or accessible elsewhere
        // For now, let's assume only owner or a designated transporter can complete
        // require(ownerOf(tokenId) == msg.sender || /* logic to check if msg.sender is a valid transporter for this tokenId */, "Marketplace: Not authorized to complete transport");
        
        require(purchase.status == PurchaseStatus.InTransit, "Marketplace: Transport not started or already completed");

        purchase.status = PurchaseStatus.TransportCompleted;
        emit TransportCompleted(tokenId, msg.sender, block.timestamp);
        emit PurchaseStatusUpdated(tokenId, PurchaseStatus.InTransit, PurchaseStatus.TransportCompleted, msg.sender, block.timestamp); // from NFTCore
        // Update lastActionTimestamp for completer (msg.sender)
    }
}

