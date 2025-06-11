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

# Bilinear interpolation function
def bilinear_interpolate(matrix, x, y):
    """
    Perform bilinear interpolation at position (x, y)
    where x, y are in continuous coordinates
    """
    h, w = matrix.shape
    
    # Get integer coordinates
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, w - 1)
    y2 = min(y1 + 1, h - 1)
    
    # Get fractional parts
    fx = x - x1
    fy = y - y1
    
    # Get corner values
    Q11 = matrix[y1, x1]
    Q12 = matrix[y2, x1]
    Q21 = matrix[y1, x2]
    Q22 = matrix[y2, x2]
    
    # Bilinear interpolation
    R1 = Q11 * (1 - fx) + Q21 * fx
    R2 = Q12 * (1 - fx) + Q22 * fx
    P = R1 * (1 - fy) + R2 * fy
    
    return P

# Create 4x4 output matrix using bilinear interpolation
output_matrix = np.zeros((4, 4))
scale_factor = 2.0

for y_out in range(4):
    for x_out in range(4):
        # Map 4x4 coordinates to 2x2 continuous coordinates
        x_in = (x_out + 0.5) / scale_factor - 0.5
        y_in = (y_out + 0.5) / scale_factor - 0.5
        
        # Clamp coordinates to valid range
        x_in = max(0, min(x_in, 1))
        y_in = max(0, min(y_in, 1))
        
        # Perform bilinear interpolation
        output_matrix[y_out, x_out] = bilinear_interpolate(input_matrix, x_in, y_in)

print(f"\nBilinear Interpolated 4x4 Output Matrix:")
print(output_matrix.astype(int))

# Convert grayscale values to quantum rotation angles for output matrix
max_value = 255
theta_output = {}
for y in range(4):
    for x in range(4):
        theta_output[(y, x)] = (output_matrix[y, x] / max_value) * (pi / 2)

print("\nRotation angles for bilinear interpolated values:")
for y in range(4):
    for x in range(4):
        print(f"Position ({y},{x}): {theta_output[(y,x)]:.4f} radians (value: {output_matrix[y,x]:.0f})")

# Create quantum circuit for 4x4 output (requires 2 qubits each for x and y coordinates)
qreg_x_prime = QuantumRegister(2, 'x_prime')  # 2 qubits for 4 x positions
qreg_y_prime = QuantumRegister(2, 'y_prime')  # 2 qubits for 4 y positions
qreg_anc = QuantumRegister(4, 'anc')          # 4 ancilla qubits for position encoding
qreg_color = QuantumRegister(1, 'color')      # 1 qubit for color/grayscale value
creg_c = ClassicalRegister(1, 'c')            # Classical register for measurement

circuit = QuantumCircuit(qreg_x_prime, qreg_y_prime, qreg_anc, qreg_color, creg_c)

# Step 1: Create superposition for all positions (4x4 = 16 positions)
circuit.h(qreg_x_prime[0])  # Create superposition for x coordinate bit 0
circuit.h(qreg_x_prime[1])  # Create superposition for x coordinate bit 1
circuit.h(qreg_y_prime[0])  # Create superposition for y coordinate bit 0
circuit.h(qreg_y_prime[1])  # Create superposition for y coordinate bit 1

# Step 2: Position encoding using ancilla qubits
# Encode x coordinate (0-3) using 2 qubits
circuit.cx(qreg_x_prime[0], qreg_anc[0])
circuit.cx(qreg_x_prime[1], qreg_anc[1])

# Encode y coordinate (0-3) using 2 qubits  
circuit.cx(qreg_y_prime[0], qreg_anc[2])
circuit.cx(qreg_y_prime[1], qreg_anc[3])

# Step 3: Apply rotations for each position in the 4x4 output matrix
# We need to create controlled rotations for each of the 16 positions

def apply_position_rotation(circuit, y, x, theta_val, qreg_anc, qreg_color):
    """Apply rotation for a specific position (y,x) with given theta value"""
    if theta_val == 0:
        return  # Skip if no rotation needed
    
    # Convert position to binary representation
    x_bits = [(x >> i) & 1 for i in range(2)]  # x in 2-bit binary
    y_bits = [(y >> i) & 1 for i in range(2)]  # y in 2-bit binary
    
    # Apply X gates to set up the correct control state
    control_qubits = []
    
    # Set up x coordinate controls
    if x_bits[0] == 0:
        circuit.x(qreg_anc[0])
    if x_bits[1] == 0:
        circuit.x(qreg_anc[1])
    
    # Set up y coordinate controls  
    if y_bits[0] == 0:
        circuit.x(qreg_anc[2])
    if y_bits[1] == 0:
        circuit.x(qreg_anc[3])
    
    # Apply multi-controlled rotation
    circuit.mcry(2 * theta_val, [qreg_anc[0], qreg_anc[1], qreg_anc[2], qreg_anc[3]], qreg_color[0])
    
    # Restore ancilla qubits
    if x_bits[0] == 0:
        circuit.x(qreg_anc[0])
    if x_bits[1] == 0:
        circuit.x(qreg_anc[1])
    if y_bits[0] == 0:
        circuit.x(qreg_anc[2])
    if y_bits[1] == 0:
        circuit.x(qreg_anc[3])

# Apply rotations for all 16 positions
for y in range(4):
    for x in range(4):
        apply_position_rotation(circuit, y, x, theta_output[(y, x)], qreg_anc, qreg_color)

# Step 4: Measure the color qubit
circuit.measure(qreg_color[0], creg_c[0])

print(f"\nQuantum Circuit for Bilinear Upscaling:")
print(circuit.draw(output='text'))

# Simulate the circuit
simulator = AerSimulator()
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=8192)
result = job.result()
counts = result.get_counts()

print(f"\nMeasurement counts: {counts}")

# Visualize input and bilinear interpolated output matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

# Bilinear interpolated output matrix
im2 = ax2.imshow(output_matrix, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Bilinear Interpolated 4x4 Matrix')
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

# Print comparison between nearest neighbor and bilinear interpolation
nearest_neighbor = np.zeros((4, 4))
for y_out in range(4):
    for x_out in range(4):
        y_in = y_out // 2
        x_in = x_out // 2
        nearest_neighbor[y_out, x_out] = input_matrix[y_in, x_in]

print(f"\nComparison:")
print(f"Nearest Neighbor 4x4 Matrix:")
print(nearest_neighbor.astype(int))
print(f"\nBilinear Interpolated 4x4 Matrix:")
print(output_matrix.astype(int))

print(f"\nBilinear interpolation provides smoother gradients between pixel values,")
print(f"creating more natural-looking upscaled images compared to nearest neighbor.")

# Print rotation angles summary
print(f"\nQuantum rotation angles for bilinear interpolation:")
for y in range(4):
    row_str = ""
    for x in range(4):
        row_str += f"{theta_output[(y,x)]:.3f} "
    print(f"Row {y}: {row_str}")