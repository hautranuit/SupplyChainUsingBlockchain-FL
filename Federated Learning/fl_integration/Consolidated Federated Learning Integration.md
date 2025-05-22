# Consolidated Federated Learning Integration

This directory contains the consolidated Federated Learning (FL) integration components for the ChainFLIP supply chain blockchain system. The integration has been streamlined to provide a single entry point for running all FL models.

## Key Features

- Single entry point script for all FL models
- Support for Sybil detection, batch monitoring, node behavior analysis, and dispute risk prediction
- Automatic data preparation from blockchain events
- Comprehensive logging and result tracking
- Flexible configuration options

## Usage

### Running All Models

To run all FL models in sequence:

```bash
python run_federated_learning.py --model all
```

### Running a Specific Model

To run a specific FL model:

```bash
python run_federated_learning.py --model sybil_detection
python run_federated_learning.py --model batch_monitoring
python run_federated_learning.py --model node_behavior
python run_federated_learning.py --model dispute_risk
```

### Additional Options

- `--num_clients`: Specify the number of FL clients to use (default: auto-determined)
- `--debug`: Enable debug mode with additional logging

Example:
```bash
python run_federated_learning.py --model sybil_detection --num_clients 3 --debug
```

## Workflow Integration

The consolidated FL script is designed to be integrated into the following workflow:

1. Run blockchain lifecycle scripts to generate normal activity data:
   ```bash
   npx hardhat run 01_deploy_and_configure.cjs --network amoy
   npx hardhat run 02_scenario_product_creation.cjs --network amoy
   npx hardhat run-scenario03 --network amoy --tokenid1 <TOKEN_ID_1> --cid1 <IPFS_CID_FOR_TOKEN_1> --tokenid2 <TOKEN_ID_2> --cid2 <IPFS_CID_FOR_TOKEN_2> --tokenid3 <TOKEN_ID_3> --cid3 <IPFS_CID_FOR_TOKEN_3>
   npx hardhat run 04_scenario_transport_and_ipfs.cjs --network amoy
   npx hardhat run 05_scenario_batch_processing.cjs --network amoy
   npx hardhat run 06_scenario_dispute_resolution.cjs --network amoy
   ```

2. Train the initial model with normal data:
   ```bash
   python fl_integration/run_federated_learning.py --model all
   ```

3. Run attack simulation:
   ```bash
   npx hardhat run 07_simulate_sybil_bribery_attack.cjs --network amoy
   ```

4. Retrain models with anomalous data:
   ```bash
   python fl_integration/run_federated_learning.py --model all
   ```

5. Validate results:
   ```bash
   python fl_integration/validate_integration.py
   ```

## Results

All model results are saved in the `results/` directory with the following files:
- `sybil_detection_results.json`
- `batch_monitoring_results.json`
- `node_behavior_timeseries_results.json`
- `dispute_risk_results.json`
- `fl_overall_results.json` (summary of all model runs)

## Troubleshooting

If you encounter issues:

1. Check that the context files exist:
   - `demo_context.json`
   - `sybil_attack_log.json`

2. Ensure all required Python packages are installed:
   ```bash
   pip install tensorflow tensorflow-federated web3 pandas numpy matplotlib
   ```

3. Review the log file for detailed error messages:
   ```bash
   cat ../fl_integration_run.log
   ```
