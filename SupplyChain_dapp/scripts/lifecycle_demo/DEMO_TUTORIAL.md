# ChainFLIP Supply Chain Lifecycle Demo Tutorial

This tutorial guides you through running a series of scripts to demonstrate the full lifecycle of the ChainFLIP supply chain management system. It covers product NFT minting, marketplace interactions, transport logging with simulated IPFS updates, batch processing of transactions, and dispute resolution.

**Note:** This demo excludes the Federated Learning components as per the request.

## 1. Prerequisites

1.  **Node.js and npm/yarn**: Ensure you have Node.js (v18 or later recommended) and a package manager installed.
2.  **Hardhat Project Setup**: This demo assumes you are running these scripts within the `SupplyChain_dapp` Hardhat project directory, which should already be set up with all dependencies installed (`npm install` or `yarn install`).
3.  **Multiple Accounts**: The scripts require at least 8 available accounts in your Hardhat network configuration (e.g., in `hardhat.config.cjs`) to represent different roles (Deployer/Admin, Manufacturer, Transporters, Retailer, Buyer, Arbitrator).
4.  **IPFS QR Scripts (Optional but Recommended for Full Demo)**: The `w3storage-upload-script` directory should be present as per the original project structure, especially for understanding the QR code generation/decryption flow, although the demo scripts simulate IPFS CIDs for on-chain logging.

## 2. Running the Demo Scripts

Run the following Hardhat scripts sequentially from the root of your `SupplyChain_dapp` project. Each script builds upon the state created by the previous one. It is recommended to run them on a local Hardhat network for ease of use and speed.

**Command to run a script:**
`npx hardhat run <path_to_script> --network localhost`

(Replace `localhost` with your desired network if different, e.g., `amoy` if you have configured it with sufficient funds and accounts).

**Demo Scripts Directory:** The scripts are located in `/home/ubuntu/lifecycle_demo_scripts/` in the sandbox. When you receive them, place them in a suitable location within your project, for example, `scripts/lifecycle_demo/` and adjust paths in the commands accordingly.

--- 

### Script 0: Starting the Backend Listener (Off-Chain Component)

Before running the on-chain transaction scripts, or in parallel in a separate terminal, you should start the backend event listener. This listener (from `Project/w3storage-upload-script/backendListener.js`) is designed to react to on-chain events.

1.  **Navigate to the script directory**:
    ```bash
    cd path/to/your/Project/w3storage-upload-script/
    ```
2.  **Ensure `ifps_qr.env` is ready**: Script `01_deploy_and_configure.js` will automatically update the `CONTRACT_ADDRESS` in this file. If running manually or for the first time, ensure it has placeholder values or is created.
    Example `ifps_qr.env` content (CONTRACT_ADDRESS will be updated by script 01):
    ```env
    CONTRACT_ADDRESS=
    WEB3STORAGE_TOKEN=YOUR_WEB3STORAGE_TOKEN_HERE
    PRIVATE_KEY_FOR_QR_ENCRYPTION=YOUR_32_BYTE_HEX_PRIVATE_KEY_HERE
    # Example PRIVATE_KEY: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    ```
    *You will need to provide your own `WEB3STORAGE_TOKEN` and `PRIVATE_KEY_FOR_QR_ENCRYPTION` if you intend to run the IPFS and QR functionalities fully as per the original `w3storage-upload-script` design. The demo scripts simulate IPFS CIDs for on-chain logging to simplify the core lifecycle demonstration.* 

3.  **Install dependencies for the listener** (if not done globally or project-wide):
    ```bash
    npm install ethers dotenv
    # Or any other dependencies listed in its package.json if it has one
    ```
4.  **Run the listener**:
    ```bash
    node backendListener.js
    ```

**Observe**: Keep this terminal open. You should see it connect to the network and log messages when it detects relevant events emitted by the smart contracts as you run the subsequent lifecycle scripts.

--- 

### Script 1: Deploy Contract & Configure Participants

This script deploys the `SupplyChainNFT` contract and configures various accounts with necessary roles (Manufacturer, Transporter, etc.) and initial reputations.

*   **Script Path (example):** `scripts/lifecycle_demo/01_deploy_and_configure.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/01_deploy_and_configure.js --network localhost`

**What to Observe:**
*   Console output showing the deployer address.
*   The deployed `SupplyChainNFT` contract address (this is crucial!).
*   Logs detailing the configuration of each participant (Manufacturer, Transporters, Retailer, Buyer, Arbitrator) with their roles, node types, and initial reputation.
*   **Crucially, this script will attempt to update the `CONTRACT_ADDRESS` in the `Project/w3storage-upload-script/ifps_qr.env` file.** Verify this update or do it manually if the script faces permission issues.
*   The backend listener (if running) might log the new contract it starts listening to if it re-initializes based on .env changes (depends on its implementation).

--- 

### Script 2: Product Creation (Minting Multiple NFTs)

This script mints several distinct product NFTs, each with unique metadata, by the configured Manufacturer.

*   **Script Path (example):** `scripts/lifecycle_demo/02_scenario_product_creation.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/02_scenario_product_creation.js --network localhost`
*   **Prerequisites:**
    - Ensure the `qr_codes` directory exists in `w3storage-upload-script/`
    - The `ifps_qr.env` file should be properly configured with valid keys
*   **What to Observe:**
*   Console output indicating it's using the contract address from the previous step (read from `demo_context.json`).
*   Logs for each NFT being minted, including its unique product ID, recipient (Manufacturer), and the resulting Token ID.
*   Validation of initial CID format (must start with 'ipfs://').
*   Automatic cleanup of any existing QR codes for the minted tokens.
*   Gas used for each minting transaction.
*   A `demo_context.json` file will be created/updated in the script's directory, storing the contract address, minted token IDs, and product details for use by subsequent scripts.
*   The backend listener might log `ProductMinted` events.

--- 

### Script 3: Marketplace & Purchase Scenarios

This script demonstrates products being listed for sale by the Manufacturer and then purchased by different Buyers. It covers initiating purchase, depositing collateral, and direct sales.

*   **Script Path (example):** `scripts/lifecycle_demo/03_scenario_marketplace_and_purchase.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/03_scenario_marketplace_and_purchase.js --network localhost`

**What to Observe:**
*   Logs for each product being listed for sale (Token ID, price).
*   Logs for buyers initiating purchases and depositing collateral.
*   Verification of NFT ownership changes after successful purchases.
*   Demonstration of a direct sale and transfer.
*   The `demo_context.json` file will be updated with new owner addresses and purchase prices.
*   The backend listener should log `ProductListedForSale`, `PurchaseInitiated`, `CollateralDeposited`, and `Transfer` (ERC721 standard) events.

--- 

### Script 4: Transport & IPFS Logging Scenarios

This script simulates the transport process for purchased products. It shows transport initiation, completion, and simulates the logging of transport data to IPFS by updating an on-chain CID field.

*   **Script Path (example):** `scripts/lifecycle_demo/04_scenario_transport_and_ipfs.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/04_scenario_transport_and_ipfs.js --network localhost`
*   **Prerequisites:**
    - Ensure the `qr_codes` directory exists in `w3storage-upload-script/`
    - The `ifps_qr.env` file should be properly configured with valid keys
    - The `demo_context.json` file should be present from previous scripts
*   **What to Observe:**
*   Validation of transporter wallet addresses before transport initiation.
*   Timestamp validation to ensure chronological order of history entries.
*   Logs for starting transport for different products (Token ID, transporters, locations).
*   Simulated IPFS CIDs being generated for transport logs.
*   Retry mechanism for IPFS gateway requests (simulated in the demo).
*   On-chain `ProductHistoryCIDUpdated` events being logged by the script (and picked up by the backend listener).
*   Logs for transport completion.
*   Automatic cleanup of old QR codes before generating new ones.
*   The `demo_context.json` file will be updated with transport CIDs, status, and last update timestamps.
*   The backend listener should log `TransportStarted`, `TransportCompleted`, and `ProductHistoryCIDUpdated` events.
*   The script will mention conceptual QR code generation points, referencing the original `w3storage-upload-script` utilities.

--- 

### Script 5: Batch Processing Scenarios

This script demonstrates the batch processing mechanism where a Secondary Node proposes a batch of transactions (e.g., final ownership transfers after transport), and Primary Nodes validate it.

*   **Script Path (example):** `scripts/lifecycle_demo/05_scenario_batch_processing.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/05_scenario_batch_processing.js --network localhost`

**What to Observe:**
*   Logs detailing the transactions being included in the proposed batch.
*   The Batch ID and the list of selected validators for the proposed batch.
*   Simulated voting by validators (approvals/denials).
*   Attempt to commit the batch and the outcome (committed or flagged).
*   If committed, verification of NFT ownership changes for batched transactions.
*   The backend listener should log `BatchProposed`, `ValidatorVoted`, and `BatchCommitted` (or `BatchFlagged`) events.

--- 

### Script 6: Dispute Resolution Scenario

This script simulates a dispute being opened for a product, the proposal and voting for arbitrator candidates, arbitrator selection, and final dispute resolution.

*   **Script Path (example):** `scripts/lifecycle_demo/06_scenario_dispute_resolution.js`
*   **Command:** `npx hardhat run scripts/lifecycle_demo/06_scenario_dispute_resolution.js --network localhost`

**What to Observe:**
*   Logs for a dispute being opened for a specific Token ID (Dispute ID, reason, evidence CID).
*   Proposal of arbitrator candidates.
*   Simulated voting for arbitrator candidates by various stakeholders.
*   Selection of the arbitrator.
*   The selected arbitrator resolving the dispute with a decision and outcome.
*   The backend listener should log `DisputeOpened`, `ArbitratorCandidateProposed`, `ArbitratorVoted`, `ArbitratorSelected`, and `DisputeResolved` events.

--- 

## 3. Resetting for a Fresh Demo

To run the demo again from scratch:

1.  **Stop the Hardhat network** if you are running it as a separate process.
2.  **Restart the Hardhat network**: `npx hardhat node` (This will give you a fresh set of accounts and a clean state).
3.  **Delete `demo_context.json`** from the directory where the scripts create it (e.g., `scripts/lifecycle_demo/demo_context.json`).
4.  **Optionally, clear the `ifps_qr.env`** `CONTRACT_ADDRESS` or restore it to a blank state if you want script 01 to populate it cleanly.
5.  **Restart the `backendListener.js`** if you stopped it.
6.  Run the scripts from `01_deploy_and_configure.js` onwards.

## 4. Tips for Recording a Demo

*   **Screen Setup**: Arrange your screen to show multiple terminal windows:
    - One for running the Hardhat scripts
    - One for the Hardhat node output (if run separately)
    - One for the `backendListener.js` output
    - One for monitoring the `qr_codes` directory (optional)
*   **Clear Explanations**: Verbally explain what each script is doing and what the expected outcomes are before running it.
*   **Highlight Key Outputs**: Point out important console logs, such as:
    - Deployed contract addresses
    - Token IDs
    - Batch IDs
    - Dispute IDs
    - IPFS CIDs (simulated)
    - Validation messages for transporter addresses and timestamps
    - QR code cleanup notifications
    - IPFS gateway retry attempts
    - Event logs from the backend listener
*   **Show `demo_context.json`**: Briefly show how `demo_context.json` is created and updated by the scripts to pass state information, including new fields like `lastUpdateTimestamp`.
*   **Pace Yourself**: Run scripts one by one, allowing time for the audience (lecturer) to absorb the information and observe the outputs.
*   **Network Choice**: Using the local Hardhat network (`--network localhost`) is generally faster and more reliable for demos. If using a testnet like Amoy, ensure your accounts are funded and be prepared for longer transaction times.
*   **Validation Points**: Highlight the new validation features:
    - Transporter address validation
    - Timestamp chronological validation
    - Initial CID format validation
    - QR code cleanup process
    - IPFS gateway retry mechanism

## 5. Troubleshooting Common Issues

*   **QR Code Cleanup Issues**:
    - If you see warnings about failed QR code cleanup, check the permissions of the `qr_codes` directory
    - Ensure the directory exists and is writable
    - The script will create the directory if it doesn't exist, but will warn if it can't

*   **IPFS Gateway Issues**:
    - If you see multiple retry attempts, check your network connection
    - The script will automatically retry up to 3 times with exponential backoff
    - If all retries fail, the script will exit with an error

*   **Timestamp Validation Issues**:
    - If you see timestamp validation errors, ensure your system clock is accurate
    - The script requires timestamps to be in chronological order
    - Each new update must have a timestamp greater than the previous one

*   **Transporter Address Issues**:
    - If you see transporter address validation errors, check the address format
    - Addresses must be valid Ethereum addresses
    - The script will validate addresses before initiating transport

This comprehensive flow should provide a clear demonstration of your project's on-chain and off-chain capabilities (excluding Federated Learning).

