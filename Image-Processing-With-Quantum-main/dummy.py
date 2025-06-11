import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

original = np.array([[0, 1], [2, 3]])

def nearest_neighbor_upscale(img):
    rows, cols = img.shape
    upscale = np.zeros((rows * 2, cols * 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            val = img[i, j]
            upscale[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = val
    return upscale

upscaled = nearest_neighbor_upscale(original)

def add_quantum_image_operations(qc, matrix, pos_reg, gray_reg):
    num_pos_qubits = len(pos_reg)
    
    for qubit in pos_reg:
        qc.h(qubit)
    
    def apply_grayscale(pos_val, gray_val):
        bin_pos = f"{pos_val:0{num_pos_qubits}b}"
        
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(pos_reg[i])
        
        gray_bin = f"{gray_val:04b}"
        for i, bit in enumerate(gray_bin):
            if bit == '1':
                if num_pos_qubits == 1:
                    qc.cx(pos_reg[0], gray_reg[i])
                else:
                    qc.mcx(pos_reg[:], gray_reg[i])
        
        for i, bit in enumerate(bin_pos):
            if bit == '0':
                qc.x(pos_reg[i])
    
    index = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            apply_grayscale(index, matrix[i][j])
            index += 1

original_pos_qubits = int(np.ceil(np.log2(original.size)))
upscaled_pos_qubits = int(np.ceil(np.log2(upscaled.size)))
max_pos_qubits = max(original_pos_qubits, upscaled_pos_qubits)
gray_qubits = 4

print(f"Original matrix:\n{original}")
print(f"Original matrix shape: {original.shape}")
print(f"Original image requires {original_pos_qubits} position qubits")

print(f"\nUpscaled matrix:\n{upscaled}")
print(f"Upscaled matrix shape: {upscaled.shape}")
print(f"Upscaled image requires {upscaled_pos_qubits} position qubits")

print(f"\nCombined circuit will use {max_pos_qubits} position qubits and {gray_qubits} grayscale qubits")

pos_reg = QuantumRegister(max_pos_qubits, name='pos')
gray_reg = QuantumRegister(gray_qubits, name='gray')
c_pos_reg = ClassicalRegister(max_pos_qubits, name='c_pos')
c_gray_reg = ClassicalRegister(gray_qubits, name='c_gray')

combined_qc = QuantumCircuit(pos_reg, gray_reg, c_pos_reg, c_gray_reg)

print("\n--- Adding Original Image Processing ---")

original_pos_subset = pos_reg[:original_pos_qubits]
add_quantum_image_operations(combined_qc, original, original_pos_subset, gray_reg)

combined_qc.measure(original_pos_subset, c_pos_reg[:original_pos_qubits])
combined_qc.measure(gray_reg, c_gray_reg)

combined_qc.barrier()

combined_qc.reset(pos_reg)
combined_qc.reset(gray_reg)

combined_qc.barrier()

print("--- Adding Upscaled Image Processing ---")

upscaled_pos_subset = pos_reg[:upscaled_pos_qubits]
add_quantum_image_operations(combined_qc, upscaled, upscaled_pos_subset, gray_reg)

combined_qc.measure(upscaled_pos_subset, c_pos_reg[:upscaled_pos_qubits])
combined_qc.measure(gray_reg, c_gray_reg)

print(f"\nCombined circuit created successfully!")
print(f"Total qubits: {combined_qc.num_qubits}")
print(f"Total classical bits: {combined_qc.num_clbits}")
print(f"Circuit depth: {combined_qc.depth()}")
print(f"Number of operations: {len(combined_qc.data)}")

def draw_combined_circuit(qc, title="Combined Quantum Circuit", filename="combined_circuit.png"):
    try:
        fig = plt.figure(figsize=(24, 14))
        
        style = {
            'backgroundcolor': '#ffffff',
            'textcolor': '#000000',
            'gatefacecolor': '#ffffff',
            'gateedgecolor': '#000000',
            'barrierfacecolor': '#ff6b6b',
            'cregbundle': True,
            'compress': False,
            'margin': [2, 0, 2, 0],
            'creglinestyle': 'solid',
            'displaytext': {
                'H': 'H',
                'X': 'X', 
                'measure': 'M',
                'reset': 'R'
            }
        }
        
        ax = qc.draw('mpl', style=style, fold=-1)
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.95)
        
        fig_width = fig.get_figwidth()
        
        plt.figtext(0.25, 0.88, 'Original Image Processing\n(2×2 matrix → 4 pixels)', 
                   fontsize=14, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.figtext(0.75, 0.88, 'Upscaled Image Processing\n(4×4 matrix → 16 pixels)', 
                   fontsize=14, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        explanation = ("This quantum circuit processes both original and upscaled images sequentially.\n"
                      "Barriers separate the sections, and reset gates clear qubits between processes.")
        plt.figtext(0.5, 0.02, explanation, fontsize=12, ha='center', style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined circuit saved as {filename}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error drawing circuit with matplotlib: {e}")
        return False

print("\n--- Visualizing Combined Quantum Circuit ---")
success = draw_combined_circuit(combined_qc, "Quantum Image Processing: Original + Upscaled (Side by Side)", "combined_quantum_circuit.png")

if not success:
    print("Matplotlib visualization failed. Trying text representation...")
    try:
        print("\nCircuit text representation:")
        print(combined_qc.draw(output='text', fold=120))
    except Exception as e:
        print(f"Text representation also failed: {e}")
        print("\nBasic circuit information:")
        print(f"Registers: {[reg.name + '[' + str(reg.size) + ']' for reg in combined_qc.qregs]}")
        print(f"Classical registers: {[reg.name + '[' + str(reg.size) + ']' for reg in combined_qc.cregs]}")

def create_individual_circuit(matrix, name):
    num_pos_qubits = int(np.ceil(np.log2(matrix.size)))
    
    pos = QuantumRegister(num_pos_qubits, name=f'pos')
    gray = QuantumRegister(4, name=f'gray')
    c_pos = ClassicalRegister(num_pos_qubits, name=f'c_pos')
    c_gray = ClassicalRegister(4, name=f'c_gray')
    
    qc = QuantumCircuit(pos, gray, c_pos, c_gray)
    add_quantum_image_operations(qc, matrix, pos, gray)
    qc.measure(pos, c_pos)
    qc.measure(gray, c_gray)
    
    return qc

print("\n--- Creating Individual Circuits for Reference ---")
qc_original_individual = create_individual_circuit(original, "Original")
qc_upscaled_individual = create_individual_circuit(upscaled, "Upscaled")

print(f"Original individual circuit: {qc_original_individual.num_qubits} qubits, depth {qc_original_individual.depth()}")
print(f"Upscaled individual circuit: {qc_upscaled_individual.num_qubits} qubits, depth {qc_upscaled_individual.depth()}")

def draw_individual_circuits(filename="individual_circuits.png"):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
        
        style_individual = {
            'backgroundcolor': '#ffffff',
            'textcolor': '#000000',
            'gatefacecolor': '#ffffff',
            'gateedgecolor': '#000000',
            'cregbundle': True,
            'margin': [1, 0, 1, 0]
        }
        
        qc_original_individual.draw('mpl', ax=ax1, style=style_individual)
        ax1.set_title('Original Matrix Circuit\n(2×2 matrix, 4 pixels, 2 pos qubits)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        qc_upscaled_individual.draw('mpl', ax=ax2, style=style_individual)
        ax2.set_title('Upscaled Matrix Circuit\n(4×4 matrix, 16 pixels, 4 pos qubits)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Individual Quantum Circuits Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Individual circuits saved as {filename}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error drawing individual circuits: {e}")
        return False

print("\n--- Visualizing Individual Circuits ---")
individual_success = draw_individual_circuits("individual_quantum_circuits.png")

if not individual_success:
    print("Individual circuit visualization failed.")
    try:
        print("\nOriginal circuit (text):")
        print(qc_original_individual.draw(output='text', fold=80))
        print("\nUpscaled circuit (text):")
        print(qc_upscaled_individual.draw(output='text', fold=80))
    except:
        print("Text representation of individual circuits also failed.")

print("\n" + "="*60)
print("QUANTUM IMAGE PROCESSING CIRCUIT SUMMARY")
print("="*60)
print(f"Original matrix: {original.shape} → {original.size} pixels")
print(f"Upscaled matrix: {upscaled.shape} → {upscaled.size} pixels")
print(f"Combined circuit: {combined_qc.num_qubits} qubits, {combined_qc.num_clbits} classical bits")
print(f"Circuit depth: {combined_qc.depth()}")
print(f"Total operations: {len(combined_qc.data)}")
print("\nThe circuit successfully demonstrates quantum image processing")
print("for both original and upscaled images in a single, continuous flow.")
print("Combined circuit saved as: combined_quantum_circuit.png")
print("Individual circuits saved as: individual_quantum_circuits.png")
print("="*60)