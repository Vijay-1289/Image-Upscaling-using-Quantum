from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt
import numpy as np

alpha = np.sqrt(3)/2
beta = complex(0.25, np.sqrt(3)/4)
norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
alpha /= norm
beta /= norm

def get_bloch_vector(state):
    a, b = state.data
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a)**2 - abs(b)**2
    theta_deg = np.degrees(np.arccos(z))
    return [x, y, z], round(theta_deg, 2)

qc_initial = QuantumCircuit(1)
qc_initial.initialize([alpha, beta], 0)
state_initial = Statevector.from_instruction(qc_initial)
vec_i, theta_i = get_bloch_vector(state_initial)


theta_rotation = np.pi/4  # change degrees in pi
qc_final = QuantumCircuit(1)
qc_final.initialize([alpha, beta], 0)
qc_final.ry(theta_rotation, 0) #Change Rotation type to your's
state_final = Statevector.from_instruction(qc_final)
vec_f, theta_f = get_bloch_vector(state_final)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_bloch_vector(vec_i, ax=ax1)
ax1.set_title(f"Initial: x={vec_i[0]:.3f}, y={vec_i[1]:.3f}, z={vec_i[2]:.3f}\nθ (from Z) = {theta_i}°")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_bloch_vector(vec_f, ax=ax2)
ax2.set_title(f"Rz(θ = {theta_rotation:.2f} rad)\nFinal: x={vec_f[0]:.3f}, y={vec_f[1]:.3f}, z={vec_f[2]:.3f}\nθ (from Z) = {theta_f}°")

plt.tight_layout()
plt.show()
