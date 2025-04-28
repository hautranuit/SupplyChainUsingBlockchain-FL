// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import "./NFTCore.sol";

abstract contract Marketplace is NFTCore {
    mapping(uint256 => uint256) public listingPrice;

    event PurchaseInitiated(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price);
    event PaymentReleaseTriggered(uint256 indexed tokenId, address indexed buyer);


    function sellProduct(uint256 tokenId, uint256 price) public virtual {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: You do not own this NFT");
        require(price > 0, "Marketplace: Price must be > 0");

        listingPrice[tokenId] = price;

        string memory history = string(abi.encodePacked(
            "LISTED_FOR_SALE;",
            "Price=",
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
        address transporter,
        string memory startLocation,
        string memory endLocation,
        uint256 distance
    ) public {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only owner can start transport");

        string memory history = string(abi.encodePacked(
            "TRANSPORT_STARTED",
            ";From=",
            startLocation,             
            ";To=",
            endLocation,               
            ";Transporter=",           
            _addressToString(transporter),
            ";Distance=",             
            uintToString(distance)     
        ));

        emit ProductHistoryUpdated(tokenId, history);
    }

    function completeTransport(uint256 tokenId) public {
        require(ownerOf(tokenId) == msg.sender, "Marketplace: Only NFT holder can complete transport");

        string memory history = string(abi.encodePacked(
            "TRANSPORT_COMPLETED",     
            ";CompletedBy=",               
            _addressToString(msg.sender)    
        ));

        emit ProductHistoryUpdated(tokenId, history);
    }
}