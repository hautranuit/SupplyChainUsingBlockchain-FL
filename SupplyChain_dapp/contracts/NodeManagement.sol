// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

abstract contract NodeManagement {
    // --- Node Management Variables ---
    mapping(address => bool) internal _verifiedNodes;
    mapping(address => uint256) public nodeReputation;
    mapping(address => uint256) public lastActionTimestamp;

    enum Role { Manufacturer, Transporter, Customer, Retailer, Arbitrator, Unassigned } // Added Unassigned
    mapping(address => Role) public roles;

    enum NodeType { Primary, Secondary, Unspecified } // Added Unspecified
    mapping(address => NodeType) public nodeTypes;

    address[] public allNodes;

    // --- Events ---
    event NodeVerified(address indexed node, bool status, uint256 timestamp);
    event ReputationUpdated(address indexed node, uint256 newReputation, int256 changeAmount, address indexed actor, string reason, uint256 timestamp);
    event NodePenalized(address indexed node, uint256 newReputation, uint256 penaltyAmount, address indexed actor, string reason, uint256 timestamp);
    event NodeRoleChanged(address indexed node, Role oldRole, Role newRole, address indexed actor, uint256 timestamp);
    event NodeTypeChanged(address indexed node, NodeType oldType, NodeType newType, address indexed actor, uint256 timestamp);

    // --- Node Management Functions ---

    // Update the verification status of a node
    function setVerifiedNode(address node, bool status) public virtual {
        _verifiedNodes[node] = status;
        lastActionTimestamp[msg.sender] = block.timestamp; // Assuming admin sets this
        lastActionTimestamp[node] = block.timestamp; // Node's own timestamp updated
        emit NodeVerified(node, status, block.timestamp);

        bool found = false;
        for (uint256 i = 0; i < allNodes.length; i++) {
            if (allNodes[i] == node) {
                found = true;
                break;
            }
        }
        if (!found) {
            allNodes.push(node); // Add node if not already present
        }
    }

    // Update the node type (Primary or Secondary)
    function setNodeType(address node, NodeType nType) public virtual {
        NodeType oldType = nodeTypes[node];
        nodeTypes[node] = nType;
        lastActionTimestamp[msg.sender] = block.timestamp; // Assuming admin sets this
        lastActionTimestamp[node] = block.timestamp;
        emit NodeTypeChanged(node, oldType, nType, msg.sender, block.timestamp);
    }

    // Update the role of a node
    function setRole(address _addr, Role _role) public virtual {
        Role oldRole = roles[_addr];
        roles[_addr] = _role;
        lastActionTimestamp[msg.sender] = block.timestamp; // Assuming admin sets this
        lastActionTimestamp[_addr] = block.timestamp;
        emit NodeRoleChanged(_addr, oldRole, _role, msg.sender, block.timestamp);
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
    function updateReputation(address node, int256 scoreChange, string memory reason) internal virtual { // Changed score to scoreChange (can be negative)
        uint256 oldReputation = nodeReputation[node];
        if (scoreChange >= 0) {
            nodeReputation[node] += uint256(scoreChange);
        } else {
            uint256 change = uint256(-scoreChange);
            if (nodeReputation[node] > change) {
                nodeReputation[node] -= change;
            } else {
                nodeReputation[node] = 0;
            }
        }
        lastActionTimestamp[node] = block.timestamp; // Actor performing action that leads to this should also update their timestamp
        emit ReputationUpdated(node, nodeReputation[node], scoreChange, msg.sender, reason, block.timestamp);
    }

    // Penalize a node, reducing its reputation if necessary
    function penalizeNode(address node, uint256 penalty, string memory reason) internal virtual { // Added reason
        uint256 oldReputation = nodeReputation[node];
        if (nodeReputation[node] > penalty) {
            nodeReputation[node] -= penalty;
        } else {
            nodeReputation[node] = 0;
        }
        lastActionTimestamp[node] = block.timestamp; // Actor performing action that leads to this should also update their timestamp
        emit NodePenalized(node, nodeReputation[node], penalty, msg.sender, reason, block.timestamp);
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
