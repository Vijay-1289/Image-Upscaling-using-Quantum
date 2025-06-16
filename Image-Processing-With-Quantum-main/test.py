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
