
import qiskit
import numpy as np
import hashlib
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Generate Quantum Random Bits
def generate_quantum_bits(num_qubits=256):
    print("Generating quantum random bits...")
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    for i in range(num_qubits):
        circuit.h(i)
        circuit.measure(i, i)
    
    simulator = Aer.get_backend("qasm_simulator")
    compiled_circuit = transpile(circuit, simulator)
    job = execute(compiled_circuit, simulator, shots=1)
    result = job.result()
    counts = result.get_counts()
    
    quantum_bits = list(counts.keys())[0]
    return quantum_bits

# Convert Quantum Bits to Cryptographic Key
def quantum_bits_to_key(quantum_bits):
    print("Converting quantum bits to cryptographic key...")
    binary_data = "".join(quantum_bits)
    key_hash = hashlib.sha256(binary_data.encode()).hexdigest()
    return key_hash

# Simulate Quantum Key Distribution (QKD)
def quantum_key_distribution():
    print("Simulating Quantum Key Distribution (QKD)...")
    alice_bits = generate_quantum_bits(128)
    bob_bits = generate_quantum_bits(128)

    key_match = sum(a == b for a, b in zip(alice_bits, bob_bits)) / len(alice_bits)
    
    print(f"Key Agreement Rate: {key_match * 100:.2f}%")
    
    if key_match > 0.95:
        print("Secure quantum key successfully established.")
        return quantum_bits_to_key(alice_bits)
    else:
        print("Key transmission error detected. Resending key...")
        return quantum_key_distribution()

# Save Quantum Key
def save_quantum_key(key):
    with open("quantum_key.txt", "w", encoding="utf-8") as f:
        f.write(key)
    print("Quantum cryptographic key saved successfully.")

if __name__ == "__main__":
    print("Starting Quantum Cryptographic Key Generation...")
    quantum_key = quantum_key_distribution()
    save_quantum_key(quantum_key)
