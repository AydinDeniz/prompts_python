
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

# Initialize quantum simulator
simulator = Aer.get_backend("aer_simulator")

# Create a quantum circuit for Grover’s Algorithm
def grovers_algorithm(n=3):
    qc = QuantumCircuit(n)

    # Apply Hadamard gates
    qc.h(range(n))

    # Oracle: Marking the target state
    qc.cz(0, 1)
    qc.cz(1, 2)

    # Apply Hadamard again
    qc.h(range(n))

    # Apply phase inversion
    qc.z(range(n))

    # Apply Hadamard again
    qc.h(range(n))

    qc.measure_all()
    return qc

# Create a quantum circuit for Shor's Algorithm
def shors_algorithm():
    qc = QuantumCircuit(4, 4)

    # Apply Hadamard to the first qubit
    qc.h(0)

    # Apply controlled unitary operations
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Apply inverse quantum Fourier transform
    qc.swap(0, 3)
    qc.h(0)
    qc.cz(0, 1)
    qc.h(1)
    qc.cz(1, 2)
    qc.h(2)

    qc.measure_all()
    return qc

# Execute a quantum circuit
def execute_quantum_circuit(qc):
    transpiled_circuit = transpile(qc, simulator)
    qobj = assemble(transpiled_circuit)
    result = simulator.run(qobj).result()
    counts = result.get_counts()
    return counts

if __name__ == "__main__":
    print("Executing Grover’s Algorithm...")
    grover_qc = grovers_algorithm()
    grover_result = execute_quantum_circuit(grover_qc)
    plot_histogram(grover_result)
    plt.title("Grover’s Algorithm Results")
    plt.show()

    print("Executing Shor’s Algorithm...")
    shor_qc = shors_algorithm()
    shor_result = execute_quantum_circuit(shor_qc)
    plot_histogram(shor_result)
    plt.title("Shor’s Algorithm Results")
    plt.show()
