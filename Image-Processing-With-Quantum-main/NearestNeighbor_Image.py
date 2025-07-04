import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import math
from numpy import pi
import cv2  # OpenCV for image display
from PIL import Image
import os

# === STEP 1: Enhanced Image Loading (TIFF, PNG, JPG, ASCII) ===
def load_image():
    # List of possible TIFF files to try
    tiff_files = [
        "Lenna_512_grey.tiff"
    ]
    # List of ASCII files to try
    ascii_files = [
        "lincon.txt", "ascii_image.txt", "image.txt"
    ]
    
    # Try TIFF files first
    for tiff_file in tiff_files:
        try:
            if os.path.exists(tiff_file):
                print(f"Loading TIFF image: {tiff_file}")
                img = Image.open(tiff_file)
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                original_img = np.array(img, dtype=np.uint8)
                print(f"Successfully loaded TIFF image of shape: {original_img.shape}")
                return original_img, f"TIFF: {tiff_file}"
        except Exception as e:
            print(f"Failed to load {tiff_file}: {e}")
            continue
    
    # Try other image formats
    # Try ASCII files
    for ascii_file in ascii_files:
        try:
            if os.path.exists(ascii_file):
                print(f"Loading ASCII file: {ascii_file}")
                with open(ascii_file, 'r') as f:
                    ascii_lines = [line.strip() for line in f if line.strip()]
                
                if not ascii_lines:
                    continue
                    
                # Convert characters to grayscale using ASCII values
                original_img = np.array([[ord(c) for c in line] for line in ascii_lines], dtype=np.uint8)
                print(f"Successfully loaded ASCII image of shape: {original_img.shape}")
                return original_img, f"ASCII: {ascii_file}"
        except Exception as e:
            print(f"Failed to load {ascii_file}: {e}")
            continue
    
    # Default fallback image
    print("No image files found. Using default 64x64 grayscale pattern.")
    original_img = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    return original_img, "Default: Generated pattern"

# Load the image
original_img, source_info = load_image()
original_shape = original_img.shape

# === STEP 2: Smart Upscaling Logic ===
def smart_upscale(img, target_size=1024):
    """
    Intelligently upscale image based on current size
    """
    current_size = max(img.shape)
    
    if current_size >= target_size:
        # If image is already large enough, optionally resize to target
        if current_size != target_size:
            zoom_factor = target_size / current_size
            upscaled = zoom(img, zoom=zoom_factor, order=1).astype(np.uint8)
            print(f"Resized from {img.shape} to {upscaled.shape}")
        else:
            upscaled = img.copy()
            print(f"Image already at target size: {img.shape}")
    else:
        # Upscale using bilinear interpolation
        zoom_factors = (target_size / img.shape[0], target_size / img.shape[1])
        upscaled = zoom(img, zoom=zoom_factors, order=1).astype(np.uint8)
        print(f"Upscaled from {img.shape} to {upscaled.shape} using zoom factors {zoom_factors}")
    
    return upscaled

# Apply smart upscaling
input_matrix = smart_upscale(original_img, target_size=64)

print(f"Source: {source_info}")
print(f"Original shape: {original_shape}")
print(f"Final processing shape: {input_matrix.shape}")

# === STEP 3: Quantum Circuit Encoding ===
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

# === STEP 4: Build and Simulate Quantum Circuit ===
circuit, (sampled_rows, sampled_cols) = create_quantum_image_circuit(input_matrix, sample_size=4)
print(f"Quantum circuit created with {circuit.num_qubits} qubits")

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
job = simulator.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()
print(f"Measurement counts: {counts}")

# === STEP 5: Enhanced Visualization ===
def display_images_matplotlib():
    """Display images using matplotlib for better control"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    im1 = axes[0,0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title(f'Original Image\n{original_shape}\n({source_info})')
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0,0])
    
    # Processed image
    im2 = axes[0,1].imshow(input_matrix, cmap='viridis', vmin=0, vmax=255)
    axes[0,1].set_title(f'Processed Image\n{input_matrix.shape}\n(Ready for Quantum)')
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[0,1])
    
    # Quantum sampling visualization
    im3 = axes[1,0].imshow(input_matrix, cmap='gray', vmin=0, vmax=255)
    # Highlight sampled pixels
    for i, row in enumerate(sampled_rows):
        for j, col in enumerate(sampled_cols):
            circle = plt.Circle((col, row), 2, color='red', fill=False, linewidth=2)
            axes[1,0].add_patch(circle)
            axes[1,0].text(col, row-5, f'{input_matrix[row, col]}', 
                          ha='center', va='center', color='yellow', fontweight='bold')
    axes[1,0].set_title('Quantum Sampled Pixels\n(Red circles)')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("y")
    
    # Quantum measurement results
    if counts:
        states = list(counts.keys())
        values = list(counts.values())
        bars = axes[1,1].bar(states, values, color=['lightblue', 'lightcoral'])
        axes[1,1].set_title('Quantum Measurements\n(1024 shots)')
        axes[1,1].set_xlabel('State')
        axes[1,1].set_ylabel('Count')
        # Add labels on bars
        for bar, val in zip(bars, values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                          str(val), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def display_images_opencv():
    """Display images using OpenCV"""
    try:
        # Convert to BGR for OpenCV display
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        processed_bgr = cv2.cvtColor(input_matrix, cv2.COLOR_GRAY2BGR)
        
        # Resize for better visibility if images are small
        display_size = 400
        if original_img.shape[0] < display_size:
            scale_factor = display_size // max(original_img.shape)
            original_display = cv2.resize(original_bgr, None, fx=scale_factor, fy=scale_factor, 
                                        interpolation=cv2.INTER_NEAREST)
        else:
            original_display = original_bgr
            
        if input_matrix.shape[0] < display_size:
            scale_factor = display_size // max(input_matrix.shape)
            processed_display = cv2.resize(processed_bgr, None, fx=scale_factor, fy=scale_factor,
                                         interpolation=cv2.INTER_NEAREST)
        else:
            processed_display = processed_bgr
        
        # Add text overlays
        cv2.putText(original_display, f'Original {original_shape}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_display, f'Processed {input_matrix.shape}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display windows
        cv2.imshow(f"Original Image - {source_info}", original_display)
        cv2.imshow("Processed for Quantum Computing", processed_display)
        
        print("Press any key to close the OpenCV windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"OpenCV display failed: {e}")
        print("Falling back to matplotlib display...")
        display_images_matplotlib()

# Choose display method
try:
    # Try OpenCV first
    display_images_opencv()
except:
    # Fallback to matplotlib
    display_images_matplotlib()

# === STEP 6: Detailed Analysis and Output ===
print("\n" + "="*70)
print("QUANTUM IMAGE PROCESSING ANALYSIS")
print("="*70)

print(f"Image Source: {source_info}")
print(f"Original Dimensions: {original_shape}")
print(f"Processed Dimensions: {input_matrix.shape}")
print(f"Data Type: {input_matrix.dtype}")

print(f"\nPixel Statistics:")
print(f"  Original - Min: {original_img.min()}, Max: {original_img.max()}, Mean: {original_img.mean():.2f}")
print(f"  Processed - Min: {input_matrix.min()}, Max: {input_matrix.max()}, Mean: {input_matrix.mean():.2f}")

print(f"\nQuantum Circuit Details:")
print(f"  Total Qubits: {circuit.num_qubits}")
print(f"  Sample Size: 4x4")
print(f"  Sampled Rows: {sampled_rows}")
print(f"  Sampled Cols: {sampled_cols}")

print(f"\nQuantum Measurements:")
total_shots = sum(counts.values()) if counts else 0
for state, count in counts.items():
    probability = count / total_shots if total_shots > 0 else 0
    print(f"  |{state}⟩: {count}/{total_shots} = {probability:.4f}")

# === STEP 7: Save Outputs ===
# Save the processed image
np.savetxt("quantum_input_image.txt", input_matrix, fmt="%d")
print(f"\nOutputs Saved:")
print(f"  quantum_input_image.txt - Processed image data")

# Optionally save as image file
try:
    processed_pil = Image.fromarray(input_matrix, mode='L')
    processed_pil.save("quantum_processed_image.png")
    print(f"  quantum_processed_image.png - Visual output")
except Exception as e:
    print(f"  Failed to save PNG: {e}")

print("="*70)