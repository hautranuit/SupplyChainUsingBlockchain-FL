// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

abstract contract NodeManagement {
    // --- Node Management Variables ---
    mapping(address => bool) internal _verifiedNodes;
    mapping(address => uint256) public nodeReputation;
    mapping(address => uint256) public lastActionTimestamp;

    enum Role { Manufacturer, Transpoter, Customer, Arbitrator }
    mapping(address => Role) public roles;

    enum NodeType { Primary, Secondary }
    mapping(address => NodeType) public nodeTypes;

    address[] public allNodes;

    // --- Events ---
    event NodeVerified(address indexed node, bool status);
    event ReputationUpdated(address indexed node, uint256 newReputation);
    event NodePenalized(address indexed node, uint256 newReputation);

    // --- Node Management Functions ---

    // Update the verification status of a node
    function setVerifiedNode(address node, bool status) public virtual {
        _verifiedNodes[node] = status;
        emit NodeVerified(node, status);
        bool found = false;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (allNodes[i] == node) {
                found = true;
                break;
            }
        }
        if (!found) {
            allNodes.push(node);
        }
    }

    // Update the node type (Primary or Secondary)
    function setNodeType(address node, NodeType nType) public virtual {
        nodeTypes[node] = nType;
    }

    // Update the role of a node
    function setRole(address _addr, Role _role) public virtual {
        roles[_addr] = _role;
    }

    // Get the total number of verified Primary nodes
    function getTotalPrimaryNodes() public view virtual returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (_verifiedNodes[allNodes[i]] && nodeTypes[allNodes[i]] == NodeType.Primary) {
                count++;
            }
        }
        return count;
    }

    // Update reputation for a node
    function updateReputation(address node, uint256 score) internal virtual {
        nodeReputation[node] += score;
        emit ReputationUpdated(node, nodeReputation[node]);
    }

    // Penalize a node, reducing its reputation if necessary
    function penalizeNode(address node, uint256 penalty) internal virtual {
        if (nodeReputation[node] > penalty) {
            nodeReputation[node] -= penalty;
        } else {
            nodeReputation[node] = 0;
        }
        emit NodePenalized(node, nodeReputation[node]);
    }

    // Check if a node is verified
    function isVerified(address node) public virtual  view returns (bool) {
        return _verifiedNodes[node];
    }

    // Check if a node is a Primary node
    function isPrimaryNode(address node) internal virtual view returns (bool) {
        return _verifiedNodes[node] && nodeTypes[node] == NodeType.Primary;
    }

    // Check if a node is a Secondary node
    function isSecondaryNode(address node) internal virtual view returns (bool) {
        return _verifiedNodes[node] && nodeTypes[node] == NodeType.Secondary;
    }

    // Get a list of all verified nodes
    function getAllVerifiedNodes() public view returns (address[] memory) {
        address[] memory verifiedNodesList = new address[](allNodes.length);
        uint256 idx = 0;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (_verifiedNodes[allNodes[i]]) {
                verifiedNodesList[idx] = allNodes[i];
                idx++;
            }
        }
        return verifiedNodesList;
    }
}
