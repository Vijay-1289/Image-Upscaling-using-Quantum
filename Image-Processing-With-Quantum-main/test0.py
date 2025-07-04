# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import Aer
# from qiskit.circuit import QuantumRegister, ClassicalRegister

# # Parameters
# bit_precision = 4  # for dx and dy in fixed-point
# intensity_bits = 4  # grayscale pixel values (0–15)

# # Quantum registers
# dx = QuantumRegister(bit_precision, 'dx')
# dy = QuantumRegister(bit_precision, 'dy')
# A = QuantumRegister(intensity_bits, 'A')  # Pixel A = 2
# B = QuantumRegister(intensity_bits, 'B')  # Pixel B = 1
# C = QuantumRegister(intensity_bits, 'C')  # Pixel C = 3
# D = QuantumRegister(intensity_bits, 'D')  # Pixel D = 4
# out = QuantumRegister(intensity_bits + 2, 'out')  # Output result (extra bits for sum)
# c_out = ClassicalRegister(intensity_bits + 2, 'c_out')  # classical bits for output

# # Create the circuit
# qc = QuantumCircuit(dx, dy, A, B, C, D, out, c_out)

# # === STEP 1: Initialize dx = dy = 0.5 (binary 0.1000)
# qc.x(dx[0])   # dx = 0.5
# qc.x(dy[0])   # dy = 0.5

# # === STEP 2: Initialize pixel values
# # A = 2 → 0010
# qc.x(A[1])
# # B = 1 → 0001
# qc.x(B[0])
# # C = 3 → 0011
# qc.x(C[0])
# qc.x(C[1])
# # D = 4 → 0100
# qc.x(D[2])

# # === STEP 3: Simulate A * 0.25 → shift right by 2
# # Just demo (A * (1-dx)(1-dy) ≈ A * 0.25)
# qc.cx(A[0], out[2])  # 2^-2 place
# qc.cx(A[1], out[3])  # 2^-3 place

# # === STEP 4: Measurement
# qc.measure(out, c_out)

# # === Transpile the circuit for optimization (no execution)
# transpiled_qc = transpile(qc, optimization_level=2)

# # Show the transpiled circuit
# print(transpiled_qc.draw(fold=120))  # wider print
import numpy as np
import matplotlib.pyplot as plt
# Qiskit Imports for quantum structure (no actual execution)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

# === Original 2x2 grayscale image ===
original = np.array([
    [2, 1],
    [3, 4]
], dtype=float)

# === Output size: 4x4 ===
new_size = (4, 4)
upscaled = np.zeros(new_size)

# Create dummy quantum structure to simulate logic
dx = QuantumRegister(4, 'dx')
dy = QuantumRegister(4, 'dy')
dummy_output = ClassicalRegister(1, 'c')
qc = QuantumCircuit(dx, dy, dummy_output)

# === Bilinear interpolation function (classical simulation, quantum-structured) ===
def bilinear_interpolate(img, new_shape):
    h_old, w_old = img.shape
    h_new, w_new = new_shape
    result = np.zeros((h_new, w_new))
    
    for i in range(h_new):
        for j in range(w_new):
            # Map new image coordinates to original image coordinates
            x = j / (w_new - 1) * (w_old - 1)
            y = i / (h_new - 1) * (h_old - 1)

            x0 = int(np.floor(x))
            x1 = min(x0 + 1, w_old - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, h_old - 1)

            dx = x - x0
            dy = y - y0

            # Get the four neighboring pixel values
            A = img[y0, x0]
            B = img[y0, x1]
            C = img[y1, x0]
            D = img[y1, x1]

            # Bilinear interpolation formula
            value = (A * (1 - dx) * (1 - dy) +
                     B * dx * (1 - dy) +
                     C * (1 - dx) * dy +
                     D * dx * dy)

            result[i, j] = value

    return result

# === Perform interpolation ===
interpolated = bilinear_interpolate(original, new_size)

# === Visualization ===
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap='gray', vmin=0, vmax=4)
plt.title('Original 2x2 Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(interpolated, cmap='gray', vmin=0, vmax=4)
plt.title('Interpolated 4x4 Image')
plt.colorbar()

plt.tight_layout()
plt.show()
