from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Define the 2x2 input matrix
input_matrix = np.array([
    [0, 63],
    [128, 255]
])

print("Input 2x2 Matrix:")
print(input_matrix)

# Convert grayscale values to quantum rotation angles
# theta_yx = (grayscale / max_value) * (pi/2)
max_value = 255  # Maximum value in our input matrix
theta = {}
for y in range(2):
    for x in range(2):
        theta[(y, x)] = (input_matrix[y, x] / max_value) * (pi / 2)

print("\nRotation angles (theta):")
for key, value in theta.items():
    print(f"Position {key}: {value:.4f} radians")

# Create quantum circuit
qreg_x_prime = QuantumRegister(2, 'x_prime')
qreg_y_prime = QuantumRegister(2, 'y_prime')
qreg_anc = QuantumRegister(2, 'anc')
qreg_color = QuantumRegister(1, 'color')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_x_prime, qreg_y_prime, qreg_anc, qreg_color, creg_c)

# Step 1: Create superposition for all positions
circuit.h(qreg_x_prime[0])  # Create superposition for x'_0
circuit.h(qreg_x_prime[1])  # Create superposition for x'_1
circuit.h(qreg_y_prime[0])  # Create superposition for y'_0  
circuit.h(qreg_y_prime[1])  # Create superposition for y'_1

# Step 2: Position mapping using CNOT gates
circuit.cx(qreg_x_prime[0], qreg_anc[0])  # x = x'_0 (map x coordinate)
circuit.cx(qreg_y_prime[0], qreg_anc[1])  # y = y'_0 (map y coordinate)

# Step 3: Color assignment for each pixel position
# We need to apply controlled rotations for each position (y,x)

# Position (0,0): value = 0, theta = 0 (no rotation needed)
# Skip this as theta = 0

# Position (0,1): value = 5, theta = 5/15 * pi/2 = pi/6
theta_01 = theta[(0, 1)]
if theta_01 > 0:
    # Set ancilla to represent position (0,1): anc[0]=1, anc[1]=0
    circuit.x(qreg_anc[0])  # Flip anc[0] to |1> for x=1
    # Apply controlled rotations when both ancilla qubits match the position
    circuit.mcry(2 * theta_01, [qreg_anc[1], qreg_anc[0]], qreg_color[0], None)
    circuit.x(qreg_anc[0])  # Restore anc[0]

# Position (1,0): value = 10, theta = 10/15 * pi/2 = pi/3
theta_10 = theta[(1, 0)]
if theta_10 > 0:
    # Set ancilla to represent position (1,0): anc[0]=0, anc[1]=1
    circuit.x(qreg_anc[1])  # Flip anc[1] to |1> for y=1
    # For position (1,0), we need anc[0]=0 and anc[1]=1
    circuit.x(qreg_anc[0])  # Temporarily flip anc[0] to make it |0> when controlled
    circuit.mcry(2 * theta_10, [qreg_anc[0], qreg_anc[1]], qreg_color[0], None)
    circuit.x(qreg_anc[0])  # Restore anc[0]
    circuit.x(qreg_anc[1])  # Restore anc[1]

# Position (1,1): value = 15, theta = 15/15 * pi/2 = pi/2
theta_11 = theta[(1, 1)]
if theta_11 > 0:
    # Set ancilla to represent position (1,1): anc[0]=1, anc[1]=1
    circuit.x(qreg_anc[0])  # Flip anc[0] to |1> for x=1
    circuit.x(qreg_anc[1])  # Flip anc[1] to |1> for y=1
    circuit.mcry(2 * theta_11, [qreg_anc[1], qreg_anc[0]], qreg_color[0], None)
    circuit.x(qreg_anc[0])  # Restore anc[0]
    circuit.x(qreg_anc[1])  # Restore anc[1]

# Alternative implementation using individual controlled rotations
# This approach applies rotations based on ancilla states without requiring CCRY

# Reset circuit for alternative approach
circuit = QuantumCircuit(qreg_x_prime, qreg_y_prime, qreg_anc, qreg_color, creg_c)

# Step 1: Create superposition
circuit.h(qreg_x_prime[0])
circuit.h(qreg_x_prime[1])
circuit.h(qreg_y_prime[0])
circuit.h(qreg_y_prime[1])

# Step 2: Position mapping
circuit.cx(qreg_x_prime[0], qreg_anc[0])
circuit.cx(qreg_y_prime[0], qreg_anc[1])

# Step 3: Apply rotations for each position using conditional logic
# Position (0,1): anc[0]=1, anc[1]=0
circuit.x(qreg_anc[0])  # Flip to |1>
circuit.cry(2 * theta[(0, 1)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=0
circuit.cry(2 * theta[(0, 1)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=1
circuit.x(qreg_anc[0])  # Restore

# Position (1,0): anc[0]=0, anc[1]=1
circuit.x(qreg_anc[1])  # Flip to |1>
circuit.cry(2 * theta[(1, 0)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=1
circuit.cry(2 * theta[(1, 0)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=0
circuit.x(qreg_anc[1])  # Restore

# Position (1,1): anc[0]=1, anc[1]=1
circuit.x(qreg_anc[0])  # Flip to |1>
circuit.x(qreg_anc[1])  # Flip to |1>
circuit.cry(2 * theta[(1, 1)], qreg_anc[1], qreg_color[0])  # Control on anc[1]=1
circuit.cry(2 * theta[(1, 1)], qreg_anc[0], qreg_color[0])  # Control on anc[0]=1
circuit.x(qreg_anc[0])  # Restore
circuit.x(qreg_anc[1])  # Restore

# Step 4: Measure the color qubit
circuit.measure(qreg_color[0], creg_c[0])

print(f"\nQuantum Circuit:")
print(circuit.draw(output='text'))

# Simulate the circuit
simulator = AerSimulator()
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=8192)
result = job.result()
counts = result.get_counts()

print(f"\nMeasurement counts: {counts}")

# Reconstruct the 4x4 output matrix
output_matrix = np.zeros((4, 4))

# For nearest neighbor upscaling from 2x2 to 4x4
for y_out in range(4):
    for x_out in range(4):
        # Map 4x4 coordinates back to 2x2 coordinates
        y_in = y_out // 2
        x_in = x_out // 2
        output_matrix[y_out, x_out] = input_matrix[y_in, x_in]

print(f"\nExpected 4x4 Output Matrix (Nearest Neighbor Upscaling):")
print(output_matrix.astype(int))

# Visualize both matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Input matrix
im1 = ax1.imshow(input_matrix, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Input 2x2 Matrix')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
# Add value labels
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(input_matrix[i, j]), ha='center', va='center', color='red', fontweight='bold')

# Output matrix
im2 = ax2.imshow(output_matrix, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Upscaled 4x4 Matrix')
ax2.set_xticks([0, 1, 2, 3])
ax2.set_yticks([0, 1, 2, 3])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# Add value labels
for i in range(4):
    for j in range(4):
        ax2.text(j, i, str(int(output_matrix[i, j])), ha='center', va='center', color='red', fontweight='bold')

plt.tight_layout()
plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)
plt.show()

print(f"\nRotation angles used:")
print(f"theta(0,0) = {theta[(0,0)]:.4f} rad (value: 0)")
print(f"theta(0,1) = {theta[(0,1)]:.4f} rad (value: 5)")
print(f"theta(1,0) = {theta[(1,0)]:.4f} rad (value: 10)")
print(f"theta(1,1) = {theta[(1,1)]:.4f} rad (value: 15)")