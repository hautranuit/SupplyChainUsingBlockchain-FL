// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./NFTCore.sol";

abstract contract Marketplace is NFTCore {
    mapping(uint256 => uint256) public listingPrice;
    // Removed: mapping(uint256 => uint256) public transportCost; // No longer calculating a fixed cost

    // Added: Mapping to store transporter involvement details for a given transport
    // tokenId => transporterIndex => transporterAddress (or some data)
    mapping(uint256 => mapping(uint256 => address)) public transportLegs;
    // Added: Mapping to store distance segments (example, could be more complex)
    mapping(uint256 => uint256[]) public distanceSegments;


    event PurchaseInitiated(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price);
    event PaymentReleaseTriggered(uint256 indexed tokenId, address indexed buyer);
    // Removed: event TransportCostCalculated(uint256 indexed tokenId, uint256 cost);

    // Removed: Internal function _calculateTransportCost

    function sellProduct(uint256 tokenId, uint256 price) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(price > 0, "Marketplace: Price must be > 0");

        listingPrice[tokenId] = price;

        string memory history = string(abi.encodePacked(
            "LISTED_FOR_SALE;Price=",
            uintToString(price)
        ));

        emit ProductHistoryUpdated(tokenId, history);
    }

    function initiatePurchase(uint256 tokenId, string memory CIDHash) public  {
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

        listingPrice[tokenId] = 0;

        emit PurchaseInitiated(tokenId, msg.sender, currentOwner, price);
    }

    function startTransport(
        uint256 tokenId,
        address[] memory transporters, // Changed to array of transporter addresses
        string memory startLocation,
        string memory endLocation,
        uint256 distance
    ) public {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only owner can start transport");
        uint256 numTransporters = transporters.length;
        require(numTransporters > 0, "Marketplace: Must specify at least one transporter");

        // --- Gas Consumption Logic --- 
        // Store each transporter involved - increases storage writes based on numTransporters
        for (uint i = 0; i < numTransporters; i++) {
            transportLegs[tokenId][i] = transporters[i];
        }

        // Example: Store distance segments - increases storage based on distance (simplified)
        // This is a basic example; a more realistic implementation might involve more complex logic
        uint numSegments = distance / 100; // Example: one segment per 100 miles
        if (distance % 100 != 0) numSegments++;
        if (numSegments == 0 && distance > 0) numSegments = 1; // At least one segment if distance > 0
        
        delete distanceSegments[tokenId]; // Clear previous segments if any
        for (uint i = 0; i < numSegments; i++) {
            distanceSegments[tokenId].push(i); // Store placeholder data for each segment
        }
        // --- End Gas Consumption Logic ---

        // Log history (simplified transporter logging)
        string memory history = string(abi.encodePacked(
            "TRANSPORT_STARTED",
            ";From=", startLocation,
            ";To=", endLocation,
            ";FirstTransporter=", _addressToString(transporters[0]), // Log first transporter for simplicity
            ";Distance=", uintToString(distance),
            ";NumTransporters=", uintToString(numTransporters)
            // Removed CalculatedCost logging
        ));

        emit ProductHistoryUpdated(tokenId, history);
    }

    function completeTransport(uint256 tokenId) public {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only NFT holder can complete transport");

        // Optional: Clear transport data upon completion to manage storage
        // delete transportLegs[tokenId];
        // delete distanceSegments[tokenId];

        string memory history = string(abi.encodePacked(
            "TRANSPORT_COMPLETED",
            ";CompletedBy=",
            _addressToString(msg.sender)
        ));

        emit ProductHistoryUpdated(tokenId, history);
    }
}
