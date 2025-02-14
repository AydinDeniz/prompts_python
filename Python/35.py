
from web3 import Web3

# Connect to Ethereum network (using Infura or local node)
infura_url = "https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID"
web3 = Web3(Web3.HTTPProvider(infura_url))

# Smart contract ABI and address
contract_address = "0xYourSmartContractAddress"
contract_abi = [
    {
        "constant": False,
        "inputs": [{"name": "userId", "type": "string"}],
        "name": "registerIdentity",
        "outputs": [],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "userId", "type": "string"}],
        "name": "verifyIdentity",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Register identity on blockchain
def register_identity(private_key, user_id):
    account = web3.eth.account.privateKeyToAccount(private_key)
    nonce = web3.eth.getTransactionCount(account.address)
    
    txn = contract.functions.registerIdentity(user_id).buildTransaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 100000,
        "gasPrice": web3.toWei("10", "gwei")
    })

    signed_txn = web3.eth.account.signTransaction(txn, private_key)
    tx_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    return web3.toHex(tx_hash)

# Verify identity on blockchain
def verify_identity(user_id):
    return contract.functions.verifyIdentity(user_id).call()

if __name__ == "__main__":
    user_id = "user_12345"
    private_key = "YOUR_PRIVATE_KEY"

    print("Registering identity on blockchain...")
    tx_hash = register_identity(private_key, user_id)
    print(f"Transaction Hash: {tx_hash}")

    print("Verifying identity...")
    is_verified = verify_identity(user_id)
    print(f"Identity Verified: {is_verified}")
