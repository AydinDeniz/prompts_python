
import os
import json
import ipfshttpclient
from web3 import Web3

# Connect to IPFS
ipfs = ipfshttpclient.connect("/ip4/127.0.0.1/tcp/5001/http")

# Connect to Ethereum Blockchain (Infura or Local Node)
infura_url = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
web3 = Web3(Web3.HTTPProvider(infura_url))

# Smart contract ABI and address
contract_address = "0xYourSmartContractAddress"
contract_abi = [
    {
        "constant": False,
        "inputs": [{"name": "fileHash", "type": "string"}],
        "name": "storeFileHash",
        "outputs": [],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "fileHash", "type": "string"}],
        "name": "verifyFile",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Upload file to IPFS
def upload_to_ipfs(file_path):
    res = ipfs.add(file_path)
    return res["Hash"]

# Store file hash on blockchain
def store_hash_on_blockchain(private_key, file_hash):
    account = web3.eth.account.privateKeyToAccount(private_key)
    nonce = web3.eth.getTransactionCount(account.address)

    txn = contract.functions.storeFileHash(file_hash).buildTransaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 100000,
        "gasPrice": web3.toWei("10", "gwei")
    })

    signed_txn = web3.eth.account.signTransaction(txn, private_key)
    tx_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    return web3.toHex(tx_hash)

# Verify file on blockchain
def verify_file(file_hash):
    return contract.functions.verifyFile(file_hash).call()

if __name__ == "__main__":
    private_key = "YOUR_PRIVATE_KEY"
    file_path = "document.pdf"

    print("Uploading file to IPFS...")
    file_hash = upload_to_ipfs(file_path)
    print(f"File uploaded with IPFS Hash: {file_hash}")

    print("Storing file hash on blockchain...")
    tx_hash = store_hash_on_blockchain(private_key, file_hash)
    print(f"Transaction Hash: {tx_hash}")

    print("Verifying file on blockchain...")
    verification_status = verify_file(file_hash)
    print(f"File Verified: {verification_status}")
