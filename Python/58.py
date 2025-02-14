
import hashlib
import json
import random
from web3 import Web3
from solcx import compile_source

# Ethereum Blockchain Connection
INFURA_URL = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

# Solidity Smart Contract for Voting
VOTING_CONTRACT_SOURCE = '''
pragma solidity ^0.8.0;

contract Voting {
    struct Voter {
        bool voted;
        uint vote;
        bool isRegistered;
    }

    struct Candidate {
        string name;
        uint voteCount;
    }

    mapping(address => Voter) public voters;
    Candidate[] public candidates;

    address public owner;

    constructor(string[] memory candidateNames) {
        owner = msg.sender;
        for (uint i = 0; i < candidateNames.length; i++) {
            candidates.push(Candidate({
                name: candidateNames[i],
                voteCount: 0
            }));
        }
    }

    function registerVoter(address voter) public {
        require(msg.sender == owner, "Only owner can register voters.");
        require(!voters[voter].isRegistered, "Voter already registered.");
        voters[voter] = Voter({voted: false, vote: 0, isRegistered: true});
    }

    function vote(uint candidateIndex) public {
        require(voters[msg.sender].isRegistered, "Not a registered voter.");
        require(!voters[msg.sender].voted, "Already voted.");
        require(candidateIndex < candidates.length, "Invalid candidate.");

        voters[msg.sender].voted = true;
        voters[msg.sender].vote = candidateIndex;
        candidates[candidateIndex].voteCount++;
    }

    function getWinner() public view returns (string memory) {
        uint winningVoteCount = 0;
        uint winningIndex = 0;

        for (uint i = 0; i < candidates.length; i++) {
            if (candidates[i].voteCount > winningVoteCount) {
                winningVoteCount = candidates[i].voteCount;
                winningIndex = i;
            }
        }

        return candidates[winningIndex].name;
    }
}
'''

# Compile Smart Contract
def compile_contract():
    compiled_sol = compile_source(VOTING_CONTRACT_SOURCE)
    contract_id, contract_interface = compiled_sol.popitem()
    return contract_interface

# Deploy Smart Contract
def deploy_contract(owner_private_key, candidate_names):
    contract_interface = compile_contract()
    Voting = web3.eth.contract(abi=contract_interface["abi"], bytecode=contract_interface["bin"])

    account = web3.eth.account.privateKeyToAccount(owner_private_key)
    tx = Voting.constructor(candidate_names).build_transaction({
        "from": account.address,
        "nonce": web3.eth.getTransactionCount(account.address),
        "gas": 2000000,
        "gasPrice": web3.toWei("10", "gwei")
    })

    signed_tx = web3.eth.account.sign_transaction(tx, owner_private_key)
    tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)
    tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_receipt.contractAddress

# Generate Zero-Knowledge Proof for Voter Authentication
def generate_zkp(voter_id, private_key):
    salt = random.randint(1000, 9999)
    proof = hashlib.sha256(f"{voter_id}{salt}{private_key}".encode()).hexdigest()
    return proof, salt

# Verify Zero-Knowledge Proof
def verify_zkp(proof, voter_id, salt, private_key):
    return proof == hashlib.sha256(f"{voter_id}{salt}{private_key}".encode()).hexdigest()

if __name__ == "__main__":
    owner_private_key = "YOUR_OWNER_PRIVATE_KEY"
    candidates = ["Alice", "Bob", "Charlie"]

    print("Deploying Voting Smart Contract...")
    contract_address = deploy_contract(owner_private_key, candidates)
    print(f"Contract deployed at: {contract_address}")

    voter_id = "voter123"
    voter_private_key = "SECRET_VOTER_KEY"

    print("Generating Zero-Knowledge Proof...")
    proof, salt = generate_zkp(voter_id, voter_private_key)
    print(f"Proof Generated: {proof}")

    print("Verifying Proof...")
    if verify_zkp(proof, voter_id, salt, voter_private_key):
        print("Voter Verified Successfully!")
    else:
        print("Verification Failed!")
