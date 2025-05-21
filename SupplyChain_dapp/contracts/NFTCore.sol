// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
* @title NFTCore
* @dev Core contract for minting, transferring NFTs and managing RFID data, collateral, and payment release.
*/
contract NFTCore is ERC721URIStorage, AccessControl {
    uint256 internal _nextTokenId = 1;
    mapping(uint256 => string) public cidMapping; // Store the CID of ProductHistory file in IPFS

    // --- RFID Data Storage ---
    struct RFIDData {
        string uniqueProductID;
        string batchNumber;
        string manufacturingDate;
        string expirationDate;
        string productType;
        string manufacturerID;
        string quickAccessURL;
        string nftReference;
    }
    
    enum PurchaseStatus {
        Idle,             
        Listed,           
        AwaitingCollateral,
        CollateralDeposited,
        InTransit,
        TransportCompleted,
        AwaitingRelease,  
        ReceiptConfirmed,
        Complete,         
        Disputed          
    }

    struct PurchaseInfo {
        address buyer;
        address seller;
        uint256 price;
        uint256 collateral; // Added collateral field
        PurchaseStatus status;
    }

    mapping(uint256 => RFIDData) public rfidDataMapping; // Retained for potential on-chain quick checks
    mapping(uint256 => PurchaseInfo) public purchaseInfos;

    // --- Roles --- // (Roles are defined in NodeManagement.sol)
    bytes32 public constant UPDATER_ROLE = keccak256("UPDATER_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    // --- Event Declarations --- (Added timestamp and actor where appropriate)
    event ProductMinted(uint256 indexed tokenId, address indexed owner, string uniqueProductID, string batchNumber, string manufacturingDate, uint256 timestamp);
    event ProductTransferred(uint256 indexed tokenId, address indexed from, address indexed to, uint256 timestamp);
    event PaymentReleased(uint256 indexed tokenId, address indexed seller, uint256 amount, uint256 timestamp);
    event TransporterIncentivePaid(uint256 indexed tokenId, address indexed transporter, uint256 amount, uint256 timestamp);
    event DisputeInitiated(uint256 indexed tokenId, address indexed initiator, address currentOwner, uint256 timestamp); // From NFTCore, DisputeResolution has DisputeOpened
    event CIDStored(uint256 indexed tokenId, string cid, address indexed actor, uint256 timestamp); // General CID storage event
    event InitialCIDStored(uint256 indexed tokenId, string initialCID, address indexed actor, uint256 timestamp); // Specific for initial CID after minting
    event CIDToHistoryStored(uint256 indexed tokenId, string newCid, address indexed actor, uint256 timestamp); // For IPFS history updates
    event CollateralDepositedForPurchase(uint256 indexed tokenId, address indexed buyer, uint256 amount, uint256 timestamp);
    event PurchaseCompleted(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price, uint256 timestamp);
    event ReceiptConfirmed(uint256 indexed tokenId, address indexed confirmer, uint256 timestamp);
    event PaymentAndTransferCompleted(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price, uint256 timestamp);
    event PurchaseStatusUpdated(uint256 indexed tokenId, PurchaseStatus oldStatus, PurchaseStatus newStatus, address indexed actor, uint256 timestamp);


    constructor(address initialAdmin)
        ERC721("SupplyChainNFT", "SCNFT") // Updated name and symbol for clarity
    {
        _grantRole(DEFAULT_ADMIN_ROLE, initialAdmin);
        _grantRole(UPDATER_ROLE, initialAdmin); 
        _grantRole(MINTER_ROLE, initialAdmin); // Grant initial admin minter role
    }

    struct MintNFTParams {
        address recipient;
        string uniqueProductID;
        string batchNumber;
        string manufacturingDate;
        string expirationDate;
        string productType;
        string manufacturerID;
        string quickAccessURL;
        string nftReference; // Could be an initial CID or other reference
    }

    function mintNFT(
        MintNFTParams memory params
    ) public onlyRole(MINTER_ROLE) returns (uint256) {
        uint256 tokenId = _nextTokenId++;
        _safeMint(params.recipient, tokenId);

        rfidDataMapping[tokenId] = RFIDData({
            uniqueProductID: params.uniqueProductID,
            batchNumber: params.batchNumber,
            manufacturingDate: params.manufacturingDate,
            expirationDate: params.expirationDate,
            productType: params.productType,
            manufacturerID: params.manufacturerID,
            quickAccessURL: params.quickAccessURL,
            nftReference: params.nftReference
        });

        emit ProductMinted(tokenId, params.recipient, params.uniqueProductID, params.batchNumber, params.manufacturingDate, block.timestamp);
        // Update lastActionTimestamp for minter and owner
        // This requires NFTCore to inherit or have access to NodeManagement's lastActionTimestamp
        // For now, assuming this will be handled in SupplyChainNFT or by off-chain logic listening to events
        return tokenId;
    }

    // Helper function to check if a token exists
    function _exists(uint256 tokenId) internal view returns (bool) {
        return _ownerOf(tokenId) != address(0);
    }

    // Renamed for clarity: this is for the *initial* metadata CID after minting
    function storeInitialCID(uint256 tokenId, string memory cid) public onlyRole(UPDATER_ROLE) {
        require(_exists(tokenId), "NFTCore: Token does not exist");
        // require(ownerOf(tokenId) == msg.sender, "NFTCore: You do not own this NFT"); // Or specific role
        cidMapping[tokenId] = cid;
        emit InitialCIDStored(tokenId, cid, msg.sender, block.timestamp);
        // Update lastActionTimestamp for msg.sender
    }

    function storeCIDToHistory(uint256 tokenId, string memory newCid) public onlyRole(UPDATER_ROLE) {
        require(_exists(tokenId), "NFTCore: Token does not exist");
        // Potentially add ownership or specific role check if needed
        // This function is for updating the history, so the CID might change over time.
        // The actual IPFS history log would be managed off-chain, this just stores the latest root CID.
        cidMapping[tokenId] = newCid; 
        emit CIDToHistoryStored(tokenId, newCid, msg.sender, block.timestamp);
        // Update lastActionTimestamp for msg.sender
    }

    function uintToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory result = new bytes(digits);
        uint256 index = digits;
        temp = value;
        while (temp != 0) {
            result[--index] = bytes1(uint8(48 + (temp % 10)));
            temp /= 10;
        }
        return string(result);
    }

    function verifyProductAuthenticity(
        uint256 tokenId,
        string memory expectedCID, // Renamed for clarity
        address currentOwner // Expected current owner
    ) public view returns (string memory) {
        address actualOwner = ownerOf(tokenId);
        if (actualOwner != currentOwner) {
            return "Ownership Mismatch"; 
        }
        string memory storedCID = cidMapping[tokenId];
        if (bytes(storedCID).length == 0) { 
            return "Stored CID not found for this token";
        }
        if (keccak256(bytes(expectedCID)) != keccak256(bytes(storedCID))) {
            return "CID Mismatch"; 
        }
        return "Product is Authentic";
    }
    
    function _transferNFTInternal(address from, address to, uint256 tokenId) internal virtual {
        _transfer(from, to, tokenId);
        emit ProductTransferred(tokenId, from, to, block.timestamp);
        // Update lastActionTimestamp for from, to, and potentially msg.sender if it's a separate entity triggering this
    }

    function depositPurchaseCollateral(uint256 tokenId) public payable {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.AwaitingCollateral, "NFTCore: Not awaiting collateral");
        // require(msg.value == purchase.price, "NFTCore: Incorrect collateral amount"); // Commented out, collateral can be different from price
        require(msg.value > 0, "NFTCore: Collateral amount must be greater than 0"); // Ensure some collateral is sent
        require(msg.sender == purchase.buyer, "NFTCore: Only buyer can deposit collateral");

        purchase.collateral = msg.value; // Set the collateral amount
        purchase.status = PurchaseStatus.CollateralDeposited;
        emit CollateralDepositedForPurchase(tokenId, msg.sender, msg.value, block.timestamp); // Emit msg.value as collateral amount
        emit PurchaseStatusUpdated(tokenId, PurchaseStatus.AwaitingCollateral, PurchaseStatus.CollateralDeposited, msg.sender, block.timestamp);
        // Update lastActionTimestamp for buyer (msg.sender)
    }

    function triggerDispute(uint256 tokenId) internal virtual {
        // This function is typically called internally when a dispute condition is met.
        // The actual dispute opening with candidates might be in DisputeResolution contract.
        emit DisputeInitiated(tokenId, msg.sender, ownerOf(tokenId), block.timestamp);
        // Update lastActionTimestamp for msg.sender
    }

    // Rename aymentInternal, changed to internal, added actor, removed seller check
    function _releasePurchasePaymentInternal(uint256 tokenId, bool meetsIncentiveCriteria, address actor) internal { 
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.TransportCompleted, "NFTCore: Payment can only be released after delivery confirmation");
        // require(msg.sender == purchase.seller, "NFTCore: Only seller can release payment"); // REMOVED - Authorization handled by public calling function

        uint256 paymentAmount = purchase.price;
        (bool success, ) = payable(purchase.seller).call{value: paymentAmount}("");
        require(success, "NFTCore: Payment transfer to seller failed");

        emit PaymentReleased(tokenId, purchase.seller, paymentAmount, block.timestamp);

        // Incentive logic was previously removed as PurchaseInfo lacks transporter details.
        // If re-added, meetsIncentiveCriteria should be used.
        // Example:
        // if (meetsIncentiveCriteria && /* condition for incentive */) {
        //     // ... pay incentive ...
        //     // emit TransporterIncentivePaid(...);
        // }

        PurchaseStatus oldStatus = purchase.status;
        purchase.status = PurchaseStatus.Complete; 
        emit PurchaseStatusUpdated(tokenId, oldStatus, PurchaseStatus.Complete, actor, block.timestamp); // Use actor
    }

    function confirmReceipt(uint256 tokenId) public {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.TransportCompleted, "NFTCore: Transport not completed or purchase not in correct status");
        // Require that msg.sender is the buyer or an authorized party
        require(msg.sender == purchase.buyer, "NFTCore: Only buyer can confirm receipt");

        purchase.status = PurchaseStatus.ReceiptConfirmed;
        emit ReceiptConfirmed(tokenId, msg.sender, block.timestamp);
        emit PurchaseStatusUpdated(tokenId, PurchaseStatus.TransportCompleted, PurchaseStatus.ReceiptConfirmed, msg.sender, block.timestamp);
        // Update lastActionTimestamp for buyer (msg.sender)
    }

    // Renamed for clarity: this is for ongoing history updates
    function updateProductHistoryCID(uint256 tokenId, string memory newCid) public onlyRole(UPDATER_ROLE) {
        require(bytes(cidMapping[tokenId]).length > 0, "NFTCore: Initial CID not set, cannot update history");
        cidMapping[tokenId] = newCid;
        emit CIDToHistoryStored(tokenId, newCid, msg.sender, block.timestamp);
        // Update lastActionTimestamp for msg.sender
    }

    function _addressToString(address _addr) internal pure returns (string memory) {
        bytes32 value = bytes32(uint256(uint160(_addr)));
        bytes memory alphabet = "0123456789abcdef";
        bytes memory str = new bytes(42);
        str[0] = "0";
        str[1] = "x";
        for (uint i = 0; i < 20; i++) {
            str[2+i*2] = alphabet[uint(uint8(value[i + 12] >> 4))];
            str[3+i*2] = alphabet[uint(uint8(value[i + 12] & 0x0f))];
        }
        return string(str);
    }

    function confirmDeliveryAndFinalize(uint256 tokenId, bool meetsIncentiveCriteria) public {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];

        require(purchase.status == PurchaseStatus.TransportCompleted, "NFTCore: Product not yet marked as transport completed by transporter"); 
        require(msg.sender == purchase.buyer || hasRole(DEFAULT_ADMIN_ROLE, msg.sender), "NFTCore: Only buyer or admin can confirm delivery");

        address currentOwner = ownerOf(tokenId); 

        if (currentOwner != purchase.buyer) {
            // Only transfer if the buyer doesn't already own it
            require(currentOwner != address(0), "NFTCore: Invalid current owner for transfer");
            _transferNFTInternal(currentOwner, purchase.buyer, tokenId);
        }
        // If currentOwner == purchase.buyer, the token is already with the buyer, so we skip the transfer.

        _releasePurchasePaymentInternal(tokenId, meetsIncentiveCriteria, msg.sender);

        emit PaymentAndTransferCompleted(tokenId, purchase.buyer, purchase.seller, purchase.price, block.timestamp);
    }

    // New public function for sellers to release funds independently
    function sellerReleaseFunds(uint256 tokenId, bool meetsIncentiveCriteria) public {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(msg.sender == purchase.seller, "NFTCore: Only seller can release payment");
        // Ensure the status is appropriate for seller to release funds
        require(purchase.status == PurchaseStatus.TransportCompleted || purchase.status == PurchaseStatus.ReceiptConfirmed, "NFTCore: Invalid status for seller to release funds");

        _releasePurchasePaymentInternal(tokenId, meetsIncentiveCriteria, msg.sender);
    }

    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721URIStorage, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}

