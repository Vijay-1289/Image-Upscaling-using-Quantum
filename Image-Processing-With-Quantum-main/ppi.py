# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# import math
# from numpy import pi

# # === STEP 1: Read ASCII file and convert to grayscale image ===
# try:
#     with open("lincon.txt", 'r') as f:
#         ascii_lines = [line.strip() for line in f if line.strip()]

#     # Convert characters to grayscale using ASCII values
#     original_img = np.array([[ord(c) for c in line] for line in ascii_lines], dtype=np.uint8)
#     original_shape = original_img.shape

#     # Upscale to 128x128 using bilinear interpolation
#     zoom_factors = (128 / original_shape[0], 128 / original_shape[1])
#     input_matrix = zoom(original_img, zoom=zoom_factors, order=1).astype(np.uint8)

# except FileNotFoundError:
#     print("lincon.txt not found. Using default 128x128 grayscale image.")
#     original_img = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
#     input_matrix = original_img.copy()

# # === STEP 2: Quantum Circuit Encoding ===
# def create_quantum_image_circuit(matrix, sample_size=4):
#     rows, cols = matrix.shape
#     row_idx = np.linspace(0, rows - 1, sample_size, dtype=int)
#     col_idx = np.linspace(0, cols - 1, sample_size, dtype=int)
#     sampled_matrix = matrix[np.ix_(row_idx, col_idx)]

#     qy = math.ceil(math.log2(sample_size))
#     qx = math.ceil(math.log2(sample_size))

#     qreg_y = QuantumRegister(qy, 'y')
#     qreg_x = QuantumRegister(qx, 'x')
#     qreg_anc = QuantumRegister(qy + qx, 'anc')
#     qreg_color = QuantumRegister(1, 'color')
#     creg_c = ClassicalRegister(1, 'c')
#     circuit = QuantumCircuit(qreg_y, qreg_x, qreg_anc, qreg_color, creg_c)

#     for q in qreg_y:
#         circuit.h(q)
#     for q in qreg_x:
#         circuit.h(q)

#     for i in range(qy):
#         circuit.cx(qreg_y[i], qreg_anc[i])
#     for i in range(qx):
#         circuit.cx(qreg_x[i], qreg_anc[qy + i])

#     max_val = sampled_matrix.max()
#     min_val = sampled_matrix.min()

#     for y in range(sample_size):
#         for x in range(sample_size):
#             value = sampled_matrix[y, x]
#             theta = ((value - min_val) / (max_val - min_val)) * pi if max_val != min_val else 0
#             if theta > 0:
#                 controls = []
#                 for i in range(qy):
#                     if (y >> i) & 1:
#                         controls.append(qreg_anc[i])
#                     else:
#                         circuit.x(qreg_anc[i])
#                         controls.append(qreg_anc[i])
#                 for i in range(qx):
#                     if (x >> i) & 1:
#                         controls.append(qreg_anc[qy + i])
#                     else:
#                         circuit.x(qreg_anc[qy + i])
#                         controls.append(qreg_anc[qy + i])

#                 if len(controls) == 1:
#                     circuit.cry(theta, controls[0], qreg_color[0])
#                 else:
#                     circuit.mcry(theta, controls, qreg_color[0])

#                 for i in range(qy):
#                     if not ((y >> i) & 1):
#                         circuit.x(qreg_anc[i])
#                 for i in range(qx):
#                     if not ((x >> i) & 1):
#                         circuit.x(qreg_anc[qy + i])

#     circuit.measure(qreg_color[0], creg_c[0])
#     return circuit, (row_idx, col_idx)

# # === STEP 3: Build and Simulate Quantum Circuit ===
# circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(input_matrix, sample_size=4)
# print(f"Quantum circuit created with {circuit.num_qubits} qubits")

# simulator = AerSimulator()
# compiled = transpile(circuit, simulator)
# job = simulator.run(compiled, shots=1024)
# result = job.result()
# counts = result.get_counts()
# print(f"Measurement counts: {counts}")

# # === STEP 4: Visualization (Show Actual Size Difference) ===
# plt.figure(figsize=(4, 4))
# plt.imshow(original_img, cmap='gray', vmin=0, vmax=255)
# plt.title("Original ASCII Image (64x64)")
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(8, 8))  # Twice the size to reflect 128x128
# plt.imshow(input_matrix, cmap='gray', vmin=0, vmax=255)
# plt.title("Upscaled 128x128 Image")
# plt.axis('off')
# plt.show()

# # === STEP 5: Save Output ===
# np.savetxt("quantum_input_image.txt", input_matrix, fmt="%d")
# print("Saved upscaled quantum image to 'quantum_input_image.txt'")

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
from numpy import pi
import cv2  # OpenCV for image display

# === STEP 1: Read ASCII file and convert to grayscale image ===
try:
    with open("lincon.txt", 'r') as f:
        ascii_lines = [line.strip() for line in f if line.strip()]

    # Convert characters to grayscale using ASCII values
    original_img = np.array([[ord(c) for c in line] for line in ascii_lines], dtype=np.uint8)
    original_shape = original_img.shape

    # Upscale to 128x128 using bilinear interpolation
    zoom_factors = (128 / original_shape[0], 128 / original_shape[1])
    input_matrix = zoom(original_img, zoom=zoom_factors, order=1).astype(np.uint8)

except FileNotFoundError:
    print("lincon.txt not found. Using default 128x128 grayscale image.")
    original_img = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
    input_matrix = original_img.copy()

# === STEP 2: Quantum Circuit Encoding ===
def create_quantum_image_circuit(matrix, sample_size=4):
    rows, cols = matrix.shape
    row_idx = np.linspace(0, rows - 1, sample_size, dtype=int)
    col_idx = np.linspace(0, cols - 1, sample_size, dtype=int)
    sampled_matrix = matrix[np.ix_(row_idx, col_idx)]

    qy = math.ceil(math.log2(sample_size))
    qx = math.ceil(math.log2(sample_size))

    qreg_y = QuantumRegister(qy, 'y')
    qreg_x = QuantumRegister(qx, 'x')
    qreg_anc = QuantumRegister(qy + qx, 'anc')
    qreg_color = QuantumRegister(1, 'color')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_y, qreg_x, qreg_anc, qreg_color, creg_c)

    for q in qreg_y:
        circuit.h(q)
    for q in qreg_x:
        circuit.h(q)

    for i in range(qy):
        circuit.cx(qreg_y[i], qreg_anc[i])
    for i in range(qx):
        circuit.cx(qreg_x[i], qreg_anc[qy + i])

    max_val = sampled_matrix.max()
    min_val = sampled_matrix.min()

    for y in range(sample_size):
        for x in range(sample_size):
            value = sampled_matrix[y, x]
            theta = ((value - min_val) / (max_val - min_val)) * pi if max_val != min_val else 0
            if theta > 0:
                controls = []
                for i in range(qy):
                    if (y >> i) & 1:
                        controls.append(qreg_anc[i])
                    else:
                        circuit.x(qreg_anc[i])
                        controls.append(qreg_anc[i])
                for i in range(qx):
                    if (x >> i) & 1:
                        controls.append(qreg_anc[qy + i])
                    else:
                        circuit.x(qreg_anc[qy + i])
                        controls.append(qreg_anc[qy + i])

                if len(controls) == 1:
                    circuit.cry(theta, controls[0], qreg_color[0])
                else:
                    circuit.mcry(theta, controls, qreg_color[0])

                for i in range(qy):
                    if not ((y >> i) & 1):
                        circuit.x(qreg_anc[i])
                for i in range(qx):
                    if not ((x >> i) & 1):
                        circuit.x(qreg_anc[qy + i])

    circuit.measure(qreg_color[0], creg_c[0])
    return circuit, (row_idx, col_idx)

# === STEP 3: Build and Simulate Quantum Circuit ===
circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(input_matrix, sample_size=4)
print(f"Quantum circuit created with {circuit.num_qubits} qubits")

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()
print(f"Measurement counts: {counts}")

# === STEP 4: Visualization with OpenCV ===
# Convert to 3-channel BGR for display
original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
upscaled_bgr = cv2.cvtColor(input_matrix, cv2.COLOR_GRAY2BGR)

# Optional: Resize for better visibility (optional, depends on screen size)
original_resized = cv2.resize(original_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
upscaled_resized = cv2.resize(upscaled_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)

# Show both images simultaneously
cv2.imshow("Original ASCII Image (64x64)", original_resized)
cv2.imshow("Upscaled 128x128 Image", upscaled_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === STEP 5: Save Output ===
np.savetxt("quantum_input_image.txt", input_matrix, fmt="%d")
print("Saved upscaled quantum image to 'quantum_input_image.txt'")
