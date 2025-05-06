require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

//console.log("Environment variables:");
//console.log("POLYGON_AMOY_RPC:", process.env.POLYGON_AMOY_RPC ? "defined" : "undefined");
//console.log("PRIVATE_KEYS:", process.env.PRIVATE_KEYS ? "defined" : "undefined");

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
            url: process.env.POLYGON_AMOY_RPC,
            accounts: process.env.PRIVATE_KEYS.split(",").map(key => `0x${key.trim()}`),
            chainId: 80002,
            gas: 30_000_000, 
        },
    },
    mocha: { 
        timeout: 600000 
    }
};
