import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create sphere data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the sphere surface
sphere = ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.1, edgecolor='gray')

# Axes quivers
x_axis = ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
y_axis = ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
z_axis = ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

# Initial Bloch vector
bloch_vector = ax.quiver(0, 0, 0, 0, 0, 1, color='purple', arrow_length_ratio=0.1, linewidth=2)

# Axis labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_box_aspect([1, 1, 1])
plt.title("Rotating Bloch Sphere Visualization")

# Function to update the Bloch vector for animation
def update(frame):
    global bloch_vector
    for artist in ax.lines + ax.collections:
        artist.remove()

    # Re-plot sphere (to clear previous vectors cleanly)
    ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.1, edgecolor='gray')
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)

    # Calculate rotating Bloch vector
    theta = np.pi / 2
    phi = frame * np.pi / 50
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    ax.quiver(0, 0, 0, x, y, z, color='purple', arrow_length_ratio=0.1, linewidth=2)

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=100)

# Save the animation
ani.save("rotating_bloch_sphere.gif", writer='pillow', fps=10)

plt.close()  # Prevents duplicate rendering in some environments
