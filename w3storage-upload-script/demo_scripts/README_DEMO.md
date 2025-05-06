# End-to-End Demo Workflow Guide

This guide provides step-by-step instructions to set up and run a complete demo of the supply chain tracking system. It assumes you have the updated project files (provided in `Final_Project_Deliverables.zip`) and have set up your environment.

## I. Prerequisites

1.  **Node.js and npm:** Ensure Node.js (v18+ recommended) and npm are installed.
2.  **Project Files:** Unzip `Final_Project_Deliverables.zip` to a working directory. All commands below should be run from the `Project` directory within this unzipped folder (e.g., `/path/to/your/unzipped/Final_Project_Deliverables/final_deliverables/Project/`).
3.  **Polygon Amoy Testnet Account:** You need an account on the Polygon Amoy testnet with some test MATIC for gas fees.
4.  **Web3.Storage Account:** A Web3.Storage account is required to get your `KEY` (did:key:...) and `PROOF` (a CAR file upload or delegation proof) for IPFS interactions.

## II. Setup Instructions

**1. Install Dependencies:**
   Navigate to the `SupplyChain` and `IFPS_QR` directories and install npm packages:
   ```bash
   cd SupplyChain
   npm install
   cd ../IFPS_QR
   npm install
   cd .. 
   ```

**2. Configure Environment Variables:**
   -   Go to the `IFPS_QR` directory.
   -   Rename `ifps_qr.env.example` (if present) or create a new file named `ifps_qr.env`.
   -   Populate `ifps_qr.env` with your actual values:
       ```env
       POLYGON_AMOY_RPC="YOUR_POLYGON_AMOY_RPC_URL" # e.g., from Alchemy, Infura
       PRIVATE_KEY="YOUR_DEPLOYER_AND_UPDATER_PRIVATE_KEY" # Private key of the account that will deploy and update
       CONTRACT_ADDRESS="" # This will be filled after deployment
       AES_SECRET_KEY="YOUR_64_CHAR_HEX_AES_SECRET_KEY" # Generate a 32-byte hex string (64 chars)
       HMAC_SECRET_KEY="YOUR_64_CHAR_HEX_HMAC_SECRET_KEY" # Generate a 32-byte hex string (64 chars)
       KEY="YOUR_WEB3_STORAGE_KEY_DID" # Your Web3.Storage w3up key (did:key:...)
       PROOF="YOUR_WEB3_STORAGE_PROOF_BASE64_ENCODED_OR_PATH" # Path to your proof.car or base64 encoded proof
       ```
     *   **Generating AES/HMAC Keys:** You can use Node.js to generate these:
         ```javascript
         // In Node.js REPL:
         // require('crypto').randomBytes(32).toString('hex') 
         // Run this twice, once for AES_SECRET_KEY, once for HMAC_SECRET_KEY
         ```
     *   **Web3.Storage Proof:** If you have a `proof.car` file from `w3up`, you can either provide the path to it or base64 encode its content and put the string here. The scripts are currently set up to expect the `KEY` and `PROOF` as string values directly from the `.env`.

   -   Go to the `SupplyChain` directory.
   -   Rename `.env.example` (if present) or create a new file named `.env`.
   -   Populate `SupplyChain/.env` with your RPC URL and Private Key (can be the same as in `ifps_qr.env`):
       ```env
       POLYGON_AMOY_RPC_URL="YOUR_POLYGON_AMOY_RPC_URL"
       PRIVATE_KEY="YOUR_DEPLOYER_PRIVATE_KEY"
       ```

**3. Deploy Smart Contracts:**
   -   Navigate to the `SupplyChain` directory.
   -   Run the deployment script:
       ```bash
       npx hardhat run scripts/deploy.js --network amoy
       ```
   -   Note the deployed `SupplyChainNFT` contract address from the output.
   -   Update `IFPS_QR/ifps_qr.env` with this `CONTRACT_ADDRESS`.

## III. Running the Demo

For the demo, you will need two terminal windows/tabs open, both navigated to the `Project/IFPS_QR/` directory.

**Terminal 1: Start the Backend Listener**

   In the first terminal, start the backend event listener:
   ```bash
   cd IFPS_QR 
   node backendListener.js
   ```
   This script will listen for smart contract events and interact with IPFS. Keep it running throughout the demo.

**Terminal 2: Execute Demo Steps**

In the second terminal (also in `Project/IFPS_QR/`), you will run the demo step scripts. We will create these scripts in the following steps. For now, this is an outline of the intended script execution flow.

**Step 1: Mint a New Product NFT**
   - Script: `node demo_scripts/01_mint_product.js <uniqueProductID> <batchNumber> <mfgDate> <expDate> <productType> <manufacturerID> <qrAccessURL> <nftRef>`
   - This will call the `mintNFT` function on the smart contract.
   - The `backendListener.js` in Terminal 1 should detect the `ProductMinted` event, fetch RFID data, create initial metadata, upload it to IPFS, and call `storeInitialCID` on the contract.
   - **Output:** Token ID of the newly minted NFT.

**Step 2: Generate Initial QR Code for the Product**
   - Script: `node demo_scripts/02_generate_initial_qr.js <tokenId>`
   - This script will fetch the IPFS CID for the given `tokenId` from the contract.
   - It will then use `generateEncryptedQR.js` logic to create an encrypted QR code containing this CID.
   - **Output:** Path to the generated QR code image and the encrypted payload string (for simulation).

**Step 3: Simulate Transporter Scan and Update Product History**
   - Script: `node demo_scripts/03_simulate_transporter_update.js <tokenId> "<encryptedQrPayload>" "New Location City" "TransporterWalletAddress"`
   - This script will:
     1.  Use `decryptCID.js` logic to decrypt the payload and get the current IPFS history CID.
     2.  Call `recordTransportLog_modified.js` logic to append a new transport event, upload the updated history to IPFS, and call `updateProductHistoryCID` on the contract.
   - The `backendListener.js` should detect the `CIDToHistoryStored` event (or the specific event like `TransportCompleted` that triggers the history update via the listener itself).
   - **Output:** Confirmation of history update and the new IPFS CID.

**Step 4: Generate QR Code with Updated History**
   - Script: `node demo_scripts/04_generate_updated_qr.js <tokenId>`
   - Fetches the *new* IPFS CID for the `tokenId` (reflecting the transport update).
   - Generates a new encrypted QR code.
   - **Output:** Path to the new QR code image and its encrypted payload.

**Step 5: Simulate Consumer Scan to View Product History**
   - Script: `node demo_scripts/05_simulate_consumer_view.js "<encryptedQrPayloadFromStep4>"`
   - This script will:
     1.  Decrypt the QR payload to get the latest IPFS history CID.
     2.  Fetch the product history JSON from IPFS using this CID.
   - **Output:** Display the fetched product history (RFID details, full history log).

## IV. Expected Outputs & Verification

-   **Terminal 1 (`backendListener.js`):** You should see log messages for each event detected (ProductMinted, CIDStored, CIDToHistoryStored, etc.), IPFS uploads/downloads, and contract interactions.
-   **Terminal 2 (Demo Scripts):** Each script will provide specific outputs like Token IDs, QR code paths, IPFS CIDs, and fetched data.
-   **IPFS:** You can verify the CIDs by accessing them through a public IPFS gateway (e.g., `https://<CID>.ipfs.w3s.link`).
-   **QR Codes:** QR code images will be saved in `IFPS_QR/qr_codes/` (or a specified output directory).

This structure will guide the creation of the demo scripts and the final user guide.

