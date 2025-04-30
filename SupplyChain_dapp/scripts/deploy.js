const signers = await ethers.getSigners();
const deployer = signers[0];

console.log("Deploying with account:", await deployer.getAddress());

const SupplyChainNFT = await ethers.getContractFactory("SupplyChainNFT", deployer);
const contract = await SupplyChainNFT.deploy("0x032041b4b356fEE1496805DD4749f181bC736FFA");

await contract.waitForDeployment();

console.log("Deployed to:", await contract.getAddress());
console.log("Transaction hash:", contract.deploymentTransaction().hash);
