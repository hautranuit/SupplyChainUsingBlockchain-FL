// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

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
        AwaitingRelease,  
        Complete,         
        Disputed          
    }

    struct PurchaseInfo {
        address buyer;
        address seller;
        uint256 price;
        PurchaseStatus status;
    }

    mapping(uint256 => RFIDData) public rfidDataMapping; // Retained for potential on-chain quick checks
    mapping(uint256 => PurchaseInfo) public purchaseInfos;

    // --- Roles ---
    bytes32 public constant UPDATER_ROLE = keccak256("UPDATER_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE"); // Added for minting

    // --- Event Declarations ---
    event ProductMinted(uint256 indexed tokenId, address indexed owner, string uniqueProductID, string batchNumber, string manufacturingDate);
    event ProductTransferred(uint256 indexed tokenId, address indexed from, address indexed to);
    event PaymentReleased(uint256 indexed tokenId, address indexed seller, uint256 amount);
    event TransporterIncentivePaid(uint256 indexed tokenId, address indexed transporter, uint256 amount);
    event DisputeInitiated(uint256 indexed tokenId, address indexed initiator, address currentOwner);
    event CIDStored(uint256 indexed tokenId, string cid); // General CID storage event
    event InitialCIDStored(uint256 indexed tokenId, string initialCID, address indexed actor); // Specific for initial CID after minting
    event CIDToHistoryStored(uint256 indexed tokenId, string newCid); // For IPFS history updates
    event CollateralDepositedForPurchase(uint256 indexed tokenId, address indexed buyer, uint256 amount);
    event PurchaseCompleted(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price);
    event ReceiptConfirmed(uint256 indexed tokenId, address indexed confirmer);
    event PaymentAndTransferCompleted(uint256 indexed tokenId, address indexed buyer, address indexed seller, uint256 price); // New event for releasePurchasePayment


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

        emit ProductMinted(tokenId, params.recipient, params.uniqueProductID, params.batchNumber, params.manufacturingDate);
        // Backend listens for ProductMinted, uploads initial metadata (including all RFIDData) to IPFS, 
        // then calls storeInitialCID with the resulting CID.
        return tokenId;
    }

    // Renamed for clarity: this is for the *initial* metadata CID after minting
    function storeInitialCID(uint256 tokenId, string memory cid) public onlyRole(UPDATER_ROLE) { 
        require(bytes(cidMapping[tokenId]).length == 0, "NFTCore: Initial CID already stored");
        cidMapping[tokenId] = cid;
        emit CIDStored(tokenId, cid); // Keep general event for any CID storage if needed elsewhere
        emit InitialCIDStored(tokenId, cid, msg.sender); // Specific event for minting flow
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
        // Basic owner check is done by ERC721 _transfer, but explicit check can be added if needed before specific logic
        _transfer(from, to, tokenId);
        emit ProductTransferred(tokenId, from, to);
    }

    function depositPurchaseCollateral(uint256 tokenId) public payable {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.AwaitingCollateral, "NFTCore: Not awaiting collateral");
        require(purchase.buyer == msg.sender, "NFTCore: Only designated buyer can deposit");
        require(msg.value == purchase.price, "NFTCore: Deposit must match price");

        // NFT is transferred to buyer upon collateral deposit, making buyer the owner.
        // This aligns with the idea that buyer holds the NFT while payment is in escrow.
        _transferNFTInternal(purchase.seller, purchase.buyer, tokenId);

        purchase.status = PurchaseStatus.AwaitingRelease;
        emit CollateralDepositedForPurchase(tokenId, msg.sender, msg.value);
    }

    function triggerDispute(uint256 tokenId) internal virtual {
        // Ensure the caller has the right to trigger a dispute (e.g., current owner/buyer)
        require(ownerOf(tokenId) == msg.sender, "NFTCore: Only owner can trigger dispute");
        purchaseInfos[tokenId].status = PurchaseStatus.Disputed;
        emit DisputeInitiated(tokenId, msg.sender, ownerOf(tokenId));
    }

    function releasePurchasePayment(
        uint256 tokenId,
        address transporter, // For potential incentive
        bool meetsIncentiveCriteria // For transporter incentive
    ) public {
        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        require(purchase.status == PurchaseStatus.AwaitingRelease, "NFTCore: Payment not awaiting release");
        require(purchase.buyer == msg.sender, "NFTCore: Only buyer can release payment");
        require(ownerOf(tokenId) == msg.sender, "NFTCore: Caller must be current NFT owner (buyer)");

        address seller = purchase.seller;
        uint256 amountToRelease = purchase.price;

        payable(seller).transfer(amountToRelease);
        emit PaymentReleased(tokenId, seller, amountToRelease);

        if (meetsIncentiveCriteria && transporter != address(0)) {
            uint256 incentive = amountToRelease / 10; // 10% incentive
            // Ensure contract has balance if incentives are paid from a central pool, 
            // or ensure buyer's deposit covers this. For now, assume buyer's deposit is only for product price.
            // This part needs careful thought on where incentive funds come from.
            // For simplicity, we'll assume the incentive is a separate transaction or covered by other means.
            // If it were to come from the buyer's deposited amount, the initial deposit logic would need to change.
            // For now, emitting an event that an incentive *should* be paid.
            // Actual transfer would require funds.
            // payable(transporter).transfer(incentive); // This would fail if contract has no ETH other than purchase price.
            emit TransporterIncentivePaid(tokenId, transporter, incentive);
        }

        purchase.status = PurchaseStatus.Complete;
        // Emit a more comprehensive event for backend to log to IPFS
        emit PaymentAndTransferCompleted(tokenId, purchase.buyer, seller, amountToRelease);
        // Old: emit ProductHistoryUpdated(tokenId, history_string); // Removed
    }

    function confirmReceipt(uint256 tokenId) public {
        require(ownerOf(tokenId) == msg.sender, "NFTCore: Only current owner can confirm receipt.");

        PurchaseInfo storage purchase = purchaseInfos[tokenId];
        // If purchase was awaiting release, and buyer confirms receipt, it implies completion.
        if (purchase.status == PurchaseStatus.AwaitingRelease) { 
            purchase.status = PurchaseStatus.Complete;
            // Emit PurchaseCompleted here if confirmReceipt is the explicit step to complete the purchase flow
            // instead of releasePurchasePayment being the final step for status change.
            // For now, releasePurchasePayment handles the .Complete status.
        }
        emit ReceiptConfirmed(tokenId, msg.sender);
        // Backend listens for ReceiptConfirmed, uploads final history log to IPFS,
        // and calls updateProductHistoryCID with the new CID.
    }

    // Renamed for clarity: this is for ongoing history updates
    function updateProductHistoryCID(uint256 tokenId, string memory newCid) public onlyRole(UPDATER_ROLE) {
        require(bytes(cidMapping[tokenId]).length > 0, "NFTCore: Initial CID not set, cannot update history");
        cidMapping[tokenId] = newCid;
        emit CIDToHistoryStored(tokenId, newCid);
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

    function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721URIStorage, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
}

