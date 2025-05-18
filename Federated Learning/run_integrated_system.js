// Master orchestration script for integrating lifecycle_demo with FL models
// Updated for new directory structure with FL integration inside Federated Learning directory
const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration - adjusted for new directory structure
const PROJECT_ROOT = path.resolve(__dirname, '..');  // One level up from Federated Learning directory
const LIFECYCLE_DEMO_DIR = path.join(PROJECT_ROOT, 'SupplyChain_dapp/scripts/lifecycle_demo');
const FL_INTEGRATION_DIR = path.join(__dirname, 'fl_integration');  // Inside Federated Learning directory
const LOG_FILE = path.join(__dirname, 'integrated_system_run.log');

// Create FL integration directory if it doesn't exist
if (!fs.existsSync(FL_INTEGRATION_DIR)) {
  fs.mkdirSync(FL_INTEGRATION_DIR, { recursive: true });
  console.log(`Created FL integration directory: ${FL_INTEGRATION_DIR}`);
}

// Helper functions
function logMessage(message) {
  const timestamp = new Date().toISOString();
  const logEntry = `[${timestamp}] ${message}\n`;
  console.log(message);
  fs.appendFileSync(LOG_FILE, logEntry, { encoding: 'utf8' });
}

function runLifecycleScript(scriptName) {
  logMessage(`Running lifecycle script: ${scriptName}`);
  try {
    execSync(`npx hardhat run ${scriptName} --network amoy`, { 
      stdio: 'inherit',
      cwd: LIFECYCLE_DEMO_DIR
    });
    logMessage(`Successfully completed: ${scriptName}`);
    return true;
  } catch (error) {
    logMessage(`ERROR in ${scriptName}: ${error.message}`);
    return false;
  }
}

function runFLModel(modelScript, modelName) {
  logMessage(`Running FL model: ${modelName}`);
  try {
    // Convert Windows path to WSL path if needed
    const scriptPath = path.join(FL_INTEGRATION_DIR, modelScript).replace(/\\/g, '/');
    execSync(`python "${scriptPath}"`, { 
      stdio: 'inherit',
      cwd: __dirname  // Run from Federated Learning directory
    });
    logMessage(`Successfully completed FL model: ${modelName}`);
    return true;
  } catch (error) {
    logMessage(`ERROR in FL model ${modelName}: ${error.message}`);
    return false;
  }
}

function checkContextFile() {
  const contextPath = path.join(LIFECYCLE_DEMO_DIR, 'demo_context.json');
  if (!fs.existsSync(contextPath)) {
    logMessage(`Warning: Context file not found at ${contextPath}`);
    return false;
  }
  try {
    const contextData = JSON.parse(fs.readFileSync(contextPath, 'utf8'));
    logMessage(`Context file found with ${Object.keys(contextData).length} keys`);
    return true;
  } catch (error) {
    logMessage(`Error parsing context file: ${error.message}`);
    return false;
  }
}

// Main execution flow
async function main() {
  logMessage("=== Starting Integrated System Run ===");
  logMessage(`Project root: ${PROJECT_ROOT}`);
  logMessage(`Lifecycle demo directory: ${LIFECYCLE_DEMO_DIR}`);
  logMessage(`FL integration directory: ${FL_INTEGRATION_DIR}`);
  
  // Step 1: Deploy and Configure
  logMessage("\n=== PHASE 1: DEPLOYMENT AND CONFIGURATION ===");
  if (!runLifecycleScript('01_deploy_and_configure.cjs')) {
    logMessage("Critical error in deployment. Aborting.");
    return;
  }
  
  // Run Sybil Detection after node configuration
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_sybil_detection.py'))) {
    runFLModel('run_sybil_detection.py', 'Sybil Detection');
  } else {
    logMessage("Warning: Sybil Detection script not found. Skipping.");
  }
  
  // Step 2: Product Creation
  logMessage("\n=== PHASE 2: PRODUCT CREATION ===");
  if (!runLifecycleScript('02_scenario_product_creation.cjs')) {
    logMessage("Error in product creation. Aborting.");
    return;
  }
  
  // Check if context file was updated
  if (!checkContextFile()) {
    logMessage("Warning: Context file issues after product creation.");
  }
  
  // Run Node Behavior Timeseries (first update)
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
    runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Timeseries (Update 1)');
  } else {
    logMessage("Warning: Node Behavior Timeseries script not found. Skipping.");
  }
  
  // Step 3: Marketplace and Purchase
  logMessage("\n=== PHASE 3: MARKETPLACE AND PURCHASE ===");
  if (!runLifecycleScript('03_scenario_marketplace_and_purchase.cjs')) {
    logMessage("Error in marketplace scenario. Aborting.");
    return;
  }
  
  // Run Node Behavior Timeseries (second update) and initial Dispute Risk
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
    runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Timeseries (Update 2)');
  }
  
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_dispute_risk.py'))) {
    runFLModel('run_dispute_risk.py', 'Dispute Risk (Initial)');
  } else {
    logMessage("Warning: Dispute Risk script not found. Skipping.");
  }
  
  // Step 4: Transport and IPFS
  logMessage("\n=== PHASE 4: TRANSPORT AND IPFS ===");
  if (!runLifecycleScript('04_scenario_transport_and_ipfs.cjs')) {
    logMessage("Error in transport scenario. Aborting.");
    return;
  }
  
  // Run Node Behavior Timeseries (third update) and initial Batch Monitoring
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
    runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Timeseries (Update 3)');
  }
  
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_batch_monitoring.py'))) {
    runFLModel('run_batch_monitoring.py', 'Batch Monitoring (Initial)');
  } else {
    logMessage("Warning: Batch Monitoring script not found. Skipping.");
  }
  
  // Step 5: Batch Processing
  logMessage("\n=== PHASE 5: BATCH PROCESSING ===");
  if (!runLifecycleScript('05_scenario_batch_processing.cjs')) {
    logMessage("Error in batch processing. Aborting.");
    return;
  }
  
  // Run Node Behavior Timeseries (fourth update) and update Batch Monitoring
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
    runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Timeseries (Update 4)');
  }
  
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_batch_monitoring.py'))) {
    runFLModel('run_batch_monitoring.py', 'Batch Monitoring (Update)');
  }
  
  // Step 6: Dispute Resolution
  logMessage("\n=== PHASE 6: DISPUTE RESOLUTION ===");
  if (!runLifecycleScript('06_scenario_dispute_resolution.cjs')) {
    logMessage("Error in dispute resolution. Aborting.");
    return;
  }
  
  // Run final FL models
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_node_behavior_timeseries.py'))) {
    runFLModel('run_node_behavior_timeseries.py', 'Node Behavior Timeseries (Final Update)');
  }
  
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_arbitrator_bias.py'))) {
    runFLModel('run_arbitrator_bias.py', 'Arbitrator Bias');
  } else {
    logMessage("Warning: Arbitrator Bias script not found. Skipping.");
  }
  
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'run_dispute_risk.py'))) {
    runFLModel('run_dispute_risk.py', 'Dispute Risk (Update)');
  }
  
  // Generate final report
  logMessage("\n=== GENERATING FINAL REPORT ===");
  if (fs.existsSync(path.join(FL_INTEGRATION_DIR, 'generate_integration_report.js'))) {
    try {
      execSync(`node ${path.join(FL_INTEGRATION_DIR, 'generate_integration_report.js')}`, { 
        stdio: 'inherit',
        cwd: __dirname
      });
      logMessage("Final report generated successfully.");
    } catch (error) {
      logMessage(`Error generating final report: ${error.message}`);
    }
  } else {
    logMessage("Report generation script not found. Creating simple summary...");
    const summary = {
      timestamp: new Date().toISOString(),
      completed: true,
      phases: [
        "Deployment and Configuration",
        "Product Creation",
        "Marketplace and Purchase",
        "Transport and IPFS",
        "Batch Processing",
        "Dispute Resolution"
      ],
      fl_models_run: [
        "Sybil Detection",
        "Node Behavior Timeseries",
        "Dispute Risk",
        "Batch Monitoring",
        "Arbitrator Bias"
      ]
    };
    
    fs.writeFileSync(
      path.join(__dirname, 'integration_summary.json'), 
      JSON.stringify(summary, null, 2),
      { encoding: 'utf8' }
    );
    logMessage("Simple summary created at integration_summary.json");
  }
  
  logMessage("=== Integrated System Run Completed ===");
}

// Execute main function
main().catch(error => {
  logMessage(`Unhandled error in main execution: ${error.message}`);
  process.exit(1);
});
