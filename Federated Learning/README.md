# Federated Learning Integration for ChainFLIP

This directory contains the Federated Learning (FL) integration components for the ChainFLIP supply chain blockchain system. The integration enables running FL models alongside blockchain lifecycle events to analyze real-world data and provide valuable predictions.

## Directory Structure

- `fl_integration/` - Contains all FL integration scripts
  - `run_sybil_detection.py` - Script for Sybil attack detection
  - `run_batch_monitoring.py` - Script for batch processing monitoring
  - `run_node_behavior_timeseries.py` - Script for node behavior analysis
  - `run_arbitrator_bias.py` - Script for arbitrator bias analysis
  - `run_dispute_risk.py` - Script for dispute risk prediction
  - `validate_integration.py` - Script to validate the integration setup
  - `results/` - Directory for storing FL model results

- `run_integrated_system.js` - Master orchestration script
- `FL_Model/` - Core FL model implementations
  - `tff_sybil_detection/` - Sybil detection model
  - `tff_batch_monitoring/` - Batch monitoring model
  - `tff_advanced_analysis/` - Advanced analysis models

## System Requirements

1. **Node.js**: Version 14.0.0 or higher
2. **Python**: Version 3.8 or higher
3. **TensorFlow**: Version 2.5.0 or higher
4. **TensorFlow Federated**: Version 0.20.0 or higher
5. **Web3.py**: Version 5.23.0 or higher
6. **Hardhat**: Version 2.9.0 or higher

## Installation

1. **Install Node.js dependencies**:
   ```bash
   cd /path/to/project/root
   npm install
   ```

2. **Install Python dependencies**:
   ```bash
   pip install tensorflow tensorflow-federated web3 pandas numpy matplotlib
   ```

3. **Create required directories**:
   ```bash
   mkdir -p fl_integration/results
   ```

## Usage

### Running the Complete Integrated System

To run the entire integrated system from start to finish:

```bash
cd /path/to/project/root
node Federated\ Learning/run_integrated_system.js
```

This script will:
1. Run each lifecycle demo script in sequence
2. Trigger appropriate FL models at each stage
3. Log all activities and save results

### Running Individual FL Models

You can run individual FL models separately:

```bash
cd /path/to/project/root/Federated\ Learning
python fl_integration/run_sybil_detection.py
python fl_integration/run_batch_monitoring.py
python fl_integration/run_node_behavior_timeseries.py
python fl_integration/run_arbitrator_bias.py
python fl_integration/run_dispute_risk.py
```

### Validating the Integration

To check if all components are properly integrated:

```bash
cd /path/to/project/root/Federated\ Learning
python fl_integration/validate_integration.py
```

## Data Flow

The integration uses the following data flow:

1. Lifecycle demo scripts generate blockchain events and update `demo_context.json`
2. FL integration scripts read data from `demo_context.json`
3. FL models process the data and generate predictions
4. Results are saved to `fl_integration/results/`

## Cross-Platform Compatibility

This integration has been designed to work across different platforms:

- All file operations use UTF-8 encoding to ensure compatibility with non-ASCII characters
- Path handling is platform-agnostic using Node.js and Python's path utilities
- Log messages use ASCII-compatible symbols instead of Unicode-only characters

## Troubleshooting

### Context File Not Found

If you encounter "Context file not found" errors:

1. Ensure lifecycle demo scripts have been run at least once
2. Check if `demo_context.json` exists in the expected location
3. Try creating a test context file using `validate_integration.py`

### Import Errors

If you encounter "No module named..." errors:

1. Ensure all Python dependencies are installed
2. Check if the directory structure matches the expected paths in the scripts
3. Create `__init__.py` files in each module directory if needed:

```bash
touch FL_Model/tff_sybil_detection/__init__.py
touch FL_Model/tff_batch_monitoring/__init__.py
touch FL_Model/tff_advanced_analysis/__init__.py
touch FL_Model/tff_advanced_analysis/arbitrator_bias/__init__.py
touch FL_Model/tff_advanced_analysis/dispute_risk/__init__.py
touch FL_Model/tff_advanced_analysis/node_behavior_timeseries/__init__.py
```

### Blockchain Connection Issues

If you encounter blockchain connection errors:

1. Check if the blockchain network is running
2. Verify RPC URL and contract addresses in configuration files
3. Ensure you have the correct ABI files for contract interaction

## Extending the System

### Adding a New FL Model

To add a new FL model:

1. Create a new script in `fl_integration/` following the existing patterns
2. Implement the corresponding data preparation module in `FL_Model/`
3. Update `run_integrated_system.js` to call your new model at the appropriate stage

### Customizing the Lifecycle Demo

If you need to customize the lifecycle demo:

1. Modify scripts in `SupplyChain_dapp/scripts/lifecycle_demo/`
2. Ensure they still update `demo_context.json` with the required information
3. Run `validate_integration.py` to verify everything still works

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow Federated team for the FL framework
- Ethereum and Hardhat communities for blockchain development tools
- All contributors to the ChainFLIP project
