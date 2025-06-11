from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import numpy as np

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Prepare source qubit (e.g., Ry with θ = π/2 for superposition)
theta = np.pi / 2
qc.ry(theta, 0)

# Optional: Hadamard for superposition
qc.h(0)

# CNOT for copy operation
qc.cx(0, 1)

# Additional gates
qc.s(1)  # Phase gate on target
qc.h(0)  # Hadamard on source

# Measure both qubits
qc.measure([0, 1], [0, 1])

# Print the circuit
print(qc.draw())

# Simulate
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()

# Plot results
plot_histogram(counts)