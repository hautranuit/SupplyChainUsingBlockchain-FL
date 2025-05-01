    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.26;

    import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
    import "@openzeppelin/contracts/access/Ownable.sol";
    import "@openzeppelin/contracts/access/AccessControl.sol"; // Import AccessControl

    /**
    * @title NFTCore
    * @dev Core contract for minting, transferring NFTs and managing RFID data, collateral, and payment release.
    */
    contract NFTCore is ERC721URIStorage, AccessControl { // Changed from Ownable
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

        mapping(uint256 => RFIDData) public rfidDataMapping;

        // Mapping to store collateral balances.
        mapping(uint256 => PurchaseInfo) public purchaseInfos;

        // --- Roles ---
        bytes32 public constant UPDATER_ROLE = keccak256("UPDATER_ROLE");

        // --- Event Declarations ---
        event ProductMinted(address indexed owner, uint256 tokenId);
        event ProductTransferred(address indexed from, address indexed to, uint256 tokenId);
        event PaymentReleased(address indexed recipient, uint256 amount);
        event CollateralDeposited(address indexed customer, uint256 amount);
        event RefundIssued(address indexed customer, uint256 amount);
        event SaleSuccessful(uint256 tokenId, address buyer, uint256 price);
        event DisputeInitiated(uint256 indexed tokenId, address buyer, address currentOwner);
        event CIDStored(uint256 tokenId, string cid);
        event ProductHistoryUpdated(uint256 indexed tokenId, string newCid); // Changed from string history 
        event CIDToHistoryStored(uint256 tokenId, string cid);
        event CollateralDepositedForPurchase(uint256 indexed tokenId, address indexed buyer, uint256 amount);
        event PurchaseCompleted(uint256 indexed tokenId);
        event ReceiptConfirmed(uint256 indexed tokenId, address indexed confirmer); // New event

        constructor(address initialOwner)
            ERC721("NFTCore", "NFTC")
        {
            _grantRole(DEFAULT_ADMIN_ROLE, initialOwner);
            _grantRole(UPDATER_ROLE, initialOwner); // Grant initial owner updater role by default
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
            string nftReference;
        }

        function mintNFT(
            MintNFTParams memory params
        ) public onlyRole(DEFAULT_ADMIN_ROLE) returns (uint256) { // Changed from onlyOwner
            uint256 tokenId = _nextTokenId++;
            _safeMint(params.recipient, tokenId);

            // Store RFID data
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

            emit ProductMinted(params.recipient, tokenId);

            // Initial history event removed. Backend listens for ProductMinted,
            // uploads initial data (RFID, image, video) to IPFS, and calls storeCID.
            // string memory productionHistory = "Product manufactured and NFT minted."; 
            // emit ProductHistoryUpdated(tokenId, productionHistory); // Removed
            return tokenId;
        }

        function storeCID(uint256 tokenId, string memory cid) public onlyRole(DEFAULT_ADMIN_ROLE) { // Changed from onlyOwner
            cidMapping[tokenId] = cid;
            emit CIDStored(tokenId, cid); 
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
            string memory CIDHash,
            address currentOwner
        ) public view returns (string memory) {
            address actualOwner = ownerOf(tokenId);
            if (actualOwner != currentOwner) {
            return "Ownership Mismatch"; 
        }
            
            string memory storedCID = cidMapping[tokenId];
            if (bytes(storedCID).length == 0) { 
                return "Stored CID not found for this token";
            }
            if (keccak256(bytes(CIDHash)) != keccak256(bytes(storedCID))) {
            return "CID Hash Mismatch"; 
            }

            return "Product is Authentic";
        }
        
        function transferNFTCore(address from, address to, uint256 tokenId) internal virtual {
            require(ownerOf(tokenId) == from, "From address is not the owner");
            _transfer(from, to, tokenId);
            emit ProductTransferred(from, to, tokenId);
        }

        function depositPurchaseCollateral(uint256 tokenId) public payable {
            PurchaseInfo storage purchase = purchaseInfos[tokenId];

            require(purchase.status == PurchaseStatus.AwaitingCollateral, "Marketplace: Not awaiting collateral for this token");
            require(purchase.buyer == msg.sender, "Marketplace: Only the designated buyer can deposit collateral");

            require(msg.value == purchase.price, "Marketplace: Deposited amount must exactly match the agreed price");

            transferNFTCore(purchase.seller, purchase.buyer, tokenId);

            purchase.status = PurchaseStatus.AwaitingRelease;

            emit CollateralDepositedForPurchase(tokenId, msg.sender, msg.value); 
        }

        function triggerDispute(uint256 tokenId) internal virtual {
            emit DisputeInitiated(tokenId, msg.sender, ownerOf(tokenId));
        }

        function releasePurchasePayment(
            uint256 tokenId,
            address transporter, 
            bool meetsIncentiveCriteria 
        ) public {
            PurchaseInfo storage purchase = purchaseInfos[tokenId];

            require(purchase.status == PurchaseStatus.AwaitingRelease, "Marketplace: Payment is not awaiting release");
            require(purchase.buyer == msg.sender, "Marketplace: Only the buyer can release the payment");
            require(ownerOf(tokenId) == msg.sender, "Marketplace: Caller must be the current NFT owner (buyer)");

            address seller = purchase.seller;
            uint256 amountToRelease = purchase.price;

            payable(seller).transfer(amountToRelease); 
            emit PaymentReleased(seller, amountToRelease);

        /* if (meetsIncentiveCriteria) {
                uint256 incentive = amountToRelease / 10; 
                require(collateralBalance[msg.sender] >= incentive, "Insufficient collateral balance for incentive");
                collateralBalance[msg.sender] -= incentive;
                payable(transporter).transfer(incentive);
                emit PaymentReleased(transporter, incentive);
            } 
        */
            purchase.status = PurchaseStatus.Complete;
            emit PurchaseCompleted(tokenId);
            string memory history = string(abi.encodePacked(
                "SOLD_AND_TRANSFERRED", 
                ";ToBuyer=",
                _addressToString(msg.sender)
            ));
            emit ProductHistoryUpdated(tokenId, history);
        }

        function confirmReceipt(uint256 tokenId) public {
            require(ownerOf(tokenId) == msg.sender, "NFTCore: Only the current owner can confirm receipt.");

            // Optional: Update purchase status if applicable
            PurchaseInfo storage purchase = purchaseInfos[tokenId];
            if (purchase.status == PurchaseStatus.AwaitingRelease) { // Or another relevant status
                purchase.status = PurchaseStatus.Complete;
                emit PurchaseCompleted(tokenId); // Emit if this is the final step
            }

            emit ReceiptConfirmed(tokenId, msg.sender);

            // Optional: Trigger final history update via backend.
            // Backend listens for ReceiptConfirmed, uploads final history to IPFS,
            // and calls updateProductHistory with the new CID.
            // string memory history = string(abi.encodePacked("Receipt confirmed by owner: ", _addressToString(msg.sender)));
            // emit ProductHistoryUpdated(tokenId, history); // Removed direct emission
        }

        function updateProductHistory(uint256 tokenId, string memory cid) public onlyRole(UPDATER_ROLE) {
            cidMapping[tokenId] = cid;
            emit CIDToHistoryStored(tokenId, cid);
        }

        function _addressToString(address _addr) internal pure returns (string memory) {
            bytes memory addressBytes = abi.encodePacked(_addr);
            bytes memory hexBytes = "0123456789abcdef";
            bytes memory stringBytes = new bytes(42);
            stringBytes[0] = '0';
            stringBytes[1] = 'x';
            for (uint256 i = 0; i < 20; i++) {
                stringBytes[2 + i * 2] = hexBytes[uint8(addressBytes[i] >> 4)];
                stringBytes[3 + i * 2] = hexBytes[uint8(addressBytes[i] & 0x0f)];
            }
            return string(stringBytes);
        }

        // Override supportsInterface to handle both ERC721URIStorage and AccessControl
        function supportsInterface(bytes4 interfaceId) public view virtual override(ERC721URIStorage, AccessControl) returns (bool) {
            return super.supportsInterface(interfaceId);
        }
    }
