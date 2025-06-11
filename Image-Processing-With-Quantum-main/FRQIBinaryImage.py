# import numpy as np
# import matplotlib.pyplot as plt
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# from qiskit.circuit.library import RYGate
# from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
# from qiskit.quantum_info import Statevector

# # Define helper functions
# def cnri(circ, control_qubits, target_qubit, theta):
#     controls = len(control_qubits)
#     cry = RYGate(theta).control(controls)
#     circ.append(cry, control_qubits + [target_qubit])

# def frqi_pixel(circ, control_qubits, target_qubit, angle, position_idx, num_qubits):
#     bin_idx = format(position_idx, f'0{num_qubits}b')
#     for bit, qubit in enumerate(control_qubits):
#         if bin_idx[bit] == '1':
#             circ.x(qubit)
#     if angle > 0:
#         cnri(circ, control_qubits, target_qubit, angle)
#     circ.barrier(label="After_Encoding")

#     negated_theta = np.pi/2 - angle
#     if angle > 0:
#         circ.x(target_qubit)
#     if negated_theta >= 0:
#         cnri(circ, control_qubits, target_qubit, negated_theta)
#     if angle > 0:
#         circ.x(target_qubit)
#     circ.barrier(label="After_Negation")

# # Image setup
# image = np.array([[1, 0], [1, 0]])
# angles = np.pi/2 * image.flatten()
# negated_image = 1 - image

# n = image.shape[0]
# num_pixels = n * n
# q = int(np.ceil(np.log2(num_pixels)))  # Position qubits

# # Define backend/runtime
# service = QiskitRuntimeService(
#     token='88b7e01d0d72ecec68894a80350a71d8fe48257f19c7304a707a40e35c081a78f52e9b82de45041ea983b973cfd03de6dfe027c12f11a29e264b8f12644c5ab3',
#     channel='ibm_quantum'
# )
# backend = service.backend("ibm_brisbane")  # You can choose a real backend too

# negated_values = []

# with Session(backend=backend) as session:
#     sampler = Sampler()

#     for pixel_value, pos_idx in zip(image.flatten(), range(num_pixels)):
#         qr = QuantumRegister(q + 1, 'q')
#         cr = ClassicalRegister(1, 'c')
#         qc = QuantumCircuit(qr, cr)

#         control_qubits = list(range(q))
#         target_qubit = q

#         angle = angles[pos_idx]
#         frqi_pixel(qc, control_qubits, target_qubit, angle, pos_idx, q)
#         qc.measure(target_qubit, cr)

#         # Submit job
#         result = sampler.run(circuits=qc, shots=10000000).result() 
#         counts = result.quasi_dists[0].nearest_probability_distribution()

#         count_1 = counts.get(1, 0)
#         prob_1 = count_1
#         theta_prime = 2 * np.arcsin(np.sqrt(prob_1))
#         measured_negated = int((theta_prime / (np.pi/2)) + 0.5)

#         negated_values.append(measured_negated)

# # Convert result to image
# negated_result_image = np.array(negated_values).reshape(image.shape)

# # Plotting
# plt.figure(figsize=(4, 2))
# plt.imshow(negated_result_image, cmap='gray', vmin=0, vmax=1)
# plt.title("Negated Image (Quantum Runtime)")
# plt.axis('off')
# plt.savefig('negated_image_runtime.png')
# plt.close()
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Sampler # Ensure Sampler is imported
from qiskit.quantum_info import Statevector

# Define helper functions
def cnri(circ, control_qubits, target_qubit, theta):
    """
    Applies a controlled RY gate to the target qubit, controlled by multiple qubits.
    """
    # Ensure control_qubits is a list for proper concatenation
    if not isinstance(control_qubits, list):
        control_qubits = [control_qubits]
    cry = RYGate(theta).control(len(control_qubits))
    circ.append(cry, control_qubits + [target_qubit])

def frqi_pixel(circ, control_qubits, target_qubit, angle, position_idx, num_qubits):
    """
    Encodes a single pixel value into the quantum circuit using FRQI.
    """
    bin_idx = format(position_idx, f'0{num_qubits}b')
    for bit, qubit in enumerate(control_qubits):
        if bin_idx[bit] == '1':
            circ.x(qubit)
    
    if angle > 0:
        cnri(circ, control_qubits, target_qubit, angle)
    circ.barrier(label="After_Encoding")

    negated_theta = np.pi/2 - angle
    if angle > 0:
        circ.x(target_qubit)
    if negated_theta >= 0:
        cnri(circ, control_qubits, target_qubit, negated_theta)
    if angle > 0:
        circ.x(target_qubit)
    circ.barrier(label="After_Negation")

# Image setup
image = np.array([[1, 0], [1, 0]])
angles = np.pi/2 * image.flatten()
negated_image_expected = 1 - image # For comparison if needed

n = image.shape[0]
num_pixels = n * n
q = int(np.ceil(np.log2(num_pixels)))  # Position qubits

# Define backend/runtime
# Replace with your actual token and desired backend
service = QiskitRuntimeService(
    token='88b7e01d0d72ecec68894a80350a71d8fe48257f19c7304a707a40e35c081a78f52e9b82de45041ea983b973cfd03de6dfe027c12f11a29e264b8f12644c5ab3',  # ** IMPORTANT: Replace with your actual IBM Quantum API token **
    channel='ibm_quantum'
)
backend = service.backend("ibm_brisbane") # Access backend directly from service

negated_values = []

# Initialize Sampler without the 'backend' argument
sampler = Sampler() # CORRECTED LINE: No 'backend' argument here

for pixel_value, pos_idx in zip(image.flatten(), range(num_pixels)):
    qr = QuantumRegister(q + 1, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)

    control_qubits = list(range(q))
    target_qubit = q

    angle = angles[pos_idx]
    frqi_pixel(qc, control_qubits, target_qubit, angle, pos_idx, q)
    qc.measure(target_qubit, cr)

    # Submit job to the sampler, passing the backend to the run method
    job = sampler.run(qc, shots=100000, backend=backend) # CORRECTED LINE: Pass backend to run()
    result = job.result()

    # Get the quasi-probability distribution for the first circuit (and only circuit in this case)
    quasi_distribution = result.quasi_dists[0]
    prob_1 = quasi_distribution.get(1, 0) # Get probability of '1', default to 0 if not present

    theta_prime = 2 * np.arcsin(np.sqrt(prob_1))
    measured_negated = int((theta_prime / (np.pi/2)) + 0.5) # Round to nearest integer (0 or 1)

    negated_values.append(measured_negated)

# Convert result to image
negated_result_image = np.array(negated_values).reshape(image.shape)

# Plotting
plt.figure(figsize=(4, 2))
plt.imshow(negated_result_image, cmap='gray', vmin=0, vmax=1)
plt.title("Negated Image (Quantum Runtime)")
plt.axis('off')
plt.savefig('negated_image_runtime.png')
plt.close()

print("Negated Image Result:")
print(negated_result_image)