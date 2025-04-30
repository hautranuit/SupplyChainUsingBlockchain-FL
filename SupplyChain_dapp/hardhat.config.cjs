require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import(\'hardhat/config\').HardhatUserConfig */
module.exports = {
    solidity: {
        version: "0.8.28", 
        settings: {
            optimizer: {
                enabled: true,
                runs: 1000,
            },
        },
    },
    networks: {
        amoy: {
            url: process.env.AMOY_RPC_URL,
            accounts: process.env.PRIVATE_KEYS.split(",").map(key => `0x${key.trim()}`),
            chainId: 80002,
            gas: 30_000_000, 
        },
    },
    mocha: { 
        timeout: 300000 
    }
};
