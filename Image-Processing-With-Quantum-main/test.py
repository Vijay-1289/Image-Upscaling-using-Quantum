from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import math
from scipy.ndimage import zoom

# === STEP 1: Load and resize input image matrix ===
try:
    with open("lincon.txt", 'r') as f:
        lines = f.readlines()

    # Convert each character to ASCII grayscale value
    original_img = np.array([[ord(c) for c in line.strip()] for line in lines], dtype=np.uint8)
    original_shape = original_img.shape

    # Resize to 64x64 using bilinear interpolation
    zoom_factors = (64 / original_img.shape[0], 64 / original_img.shape[1])
    img = zoom(original_img, zoom=zoom_factors, order=1).astype(np.uint8)

    print(f"Original image shape: {original_shape} -> Resized to: {img.shape}")

except FileNotFoundError:
    print("lincon.txt not found, creating example 64x64 matrix")
    img = np.zeros((64, 64), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j] = (i + j) % 256

input_matrix = img
print("Input Matrix Shape:", input_matrix.shape)
print(f"Value range: {input_matrix.min()} to {input_matrix.max()}")


# === STEP 2: Quantum Encoding ===
def create_quantum_image_circuit(matrix, sample_size=4):
    rows, cols = matrix.shape

    # Sample representative pixels
    if rows > sample_size or cols > sample_size:
        row_indices = np.linspace(0, rows - 1, sample_size, dtype=int)
        col_indices = np.linspace(0, cols - 1, sample_size, dtype=int)
        sampled_matrix = matrix[np.ix_(row_indices, col_indices)]
        print(f"Sampling {sample_size}x{sample_size} from {rows}x{cols} matrix")
    else:
        sampled_matrix = matrix
        row_indices = np.arange(rows)
        col_indices = np.arange(cols)

    sample_rows, sample_cols = sampled_matrix.shape
    qubits_for_rows = max(1, math.ceil(math.log2(sample_rows)))
    qubits_for_cols = max(1, math.ceil(math.log2(sample_cols)))

    qreg_y = QuantumRegister(qubits_for_rows, 'y_prime')
    qreg_x = QuantumRegister(qubits_for_cols, 'x_prime')
    qreg_anc = QuantumRegister(qubits_for_rows + qubits_for_cols, 'anc')
    qreg_color = QuantumRegister(1, 'color')
    creg_c = ClassicalRegister(1, 'c')

    circuit = QuantumCircuit(qreg_y, qreg_x, qreg_anc, qreg_color, creg_c)

    # Superposition
    for i in range(qubits_for_rows):
        circuit.h(qreg_y[i])
    for i in range(qubits_for_cols):
        circuit.h(qreg_x[i])

    # Copy to ancilla
    for i in range(qubits_for_rows):
        circuit.cx(qreg_y[i], qreg_anc[i])
    for i in range(qubits_for_cols):
        circuit.cx(qreg_x[i], qreg_anc[qubits_for_rows + i])

    # Controlled rotations
    max_val = sampled_matrix.max()
    min_val = sampled_matrix.min()

    for y in range(sample_rows):
        for x in range(sample_cols):
            normalized_val = (sampled_matrix[y, x] - min_val) / (max_val - min_val) if max_val != min_val else 0
            theta_val = normalized_val * pi

            if theta_val > 0:
                control_qubits = []

                for bit_pos in range(qubits_for_rows):
                    if (y >> bit_pos) & 1:
                        control_qubits.append(qreg_anc[bit_pos])
                    else:
                        circuit.x(qreg_anc[bit_pos])
                        control_qubits.append(qreg_anc[bit_pos])

                for bit_pos in range(qubits_for_cols):
                    if (x >> bit_pos) & 1:
                        control_qubits.append(qreg_anc[qubits_for_rows + bit_pos])
                    else:
                        circuit.x(qreg_anc[qubits_for_rows + bit_pos])
                        control_qubits.append(qreg_anc[qubits_for_rows + bit_pos])

                if len(control_qubits) == 1:
                    circuit.cry(theta_val, control_qubits[0], qreg_color[0])
                elif len(control_qubits) > 1:
                    circuit.mcry(theta_val, control_qubits, qreg_color[0])

                for bit_pos in range(qubits_for_rows):
                    if not ((y >> bit_pos) & 1):
                        circuit.x(qreg_anc[bit_pos])
                for bit_pos in range(qubits_for_cols):
                    if not ((x >> bit_pos) & 1):
                        circuit.x(qreg_anc[qubits_for_rows + bit_pos])

    circuit.measure(qreg_color[0], creg_c[0])

    return circuit, sampled_matrix, (row_indices, col_indices)


# === STEP 3: Create Quantum Circuit ===
print("\nCreating quantum circuit...")
quantum_circuit, sampled_matrix, (sampled_rows, sampled_cols) = create_quantum_image_circuit(input_matrix, sample_size=4)
print(f"\nQuantum Circuit created with {quantum_circuit.num_qubits} qubits")
print("Circuit depth:", quantum_circuit.depth())

if quantum_circuit.num_qubits <= 15:
    print("\nQuantum Circuit:")
    print(quantum_circuit.draw(output='text', fold=120))
else:
    print(f"\nCircuit too large to display ({quantum_circuit.num_qubits} qubits)")
    print("Structure: Superposition → Address Encoding → Pixel Value Encoding → Measurement")


# === STEP 4: Simulate Circuit ===
print("\nSimulating quantum circuit...")
try:
    simulator = AerSimulator()
    compiled_circuit = transpile(quantum_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print(f"Measurement counts: {counts}")
    total_shots = sum(counts.values())
    for state, count in counts.items():
        probability = count / total_shots
        print(f"State |{state}>: {count}/{total_shots} = {probability:.3f}")
except Exception as e:
    print(f"Simulation error: {e}")


# === STEP 5: Visualization ===
print("\nCreating visualizations...")
display_size = min(16, input_matrix.shape[0])
display_matrix = input_matrix[:display_size, :display_size]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].imshow(display_matrix, cmap='gray')
axes[0].set_title(f'Original Matrix ({display_size}x{display_size} subset)')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(sampled_matrix, cmap='gray')
axes[1].set_title('Sampled Matrix (Quantum Processed)')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')
plt.colorbar(im2, ax=axes[1])

for i in range(sampled_matrix.shape[0]):
    for j in range(sampled_matrix.shape[1]):
        axes[1].text(j, i, str(sampled_matrix[i, j]),
                     ha='center', va='center', color='red', fontweight='bold')

if sampled_matrix.shape == display_matrix.shape:
    diff_matrix = np.abs(display_matrix.astype(float) - sampled_matrix.astype(float))
    im3 = axes[2].imshow(diff_matrix, cmap='hot')
    axes[2].set_title('Processing Difference')
    plt.colorbar(im3, ax=axes[2])
else:
    max_val = sampled_matrix.max()
    min_val = sampled_matrix.min()
    rotation_matrix = np.zeros_like(sampled_matrix, dtype=float)
    for i in range(sampled_matrix.shape[0]):
        for j in range(sampled_matrix.shape[1]):
            normalized = (sampled_matrix[i, j] - min_val) / (max_val - min_val) if max_val != min_val else 0
            rotation_matrix[i, j] = normalized * pi
    im3 = axes[2].imshow(rotation_matrix, cmap='viridis')
    axes[2].set_title('Rotation Angles (radians)')
    plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()


# === STEP 6: Summary ===
print(f"\nProcessing Summary:")
print(f"Original matrix shape: {input_matrix.shape}")
print(f"Sampled matrix shape: {sampled_matrix.shape}")
print(f"Value range: {input_matrix.min()} - {input_matrix.max()}")
print(f"Quantum circuit qubits: {quantum_circuit.num_qubits}")
print(f"Circuit depth: {quantum_circuit.depth()}")

# Save results
print(f"\nSaving results...")
np.savetxt('quantum_processed_sample.txt', sampled_matrix, fmt='%d')
print("Sampled matrix saved to 'quantum_processed_sample.txt'")

print(f"\nRotation angles for sampled positions:")
max_val = sampled_matrix.max()
min_val = sampled_matrix.min()
for i in range(sampled_matrix.shape[0]):
    for j in range(sampled_matrix.shape[1]):
        normalized = (sampled_matrix[i, j] - min_val) / (max_val - min_val) if max_val != min_val else 0
        theta_val = normalized * pi
        print(f"Position ({sampled_rows[i]}, {sampled_cols[j]}): "
              f"value={sampled_matrix[i, j]}, θ={theta_val:.4f} rad")
