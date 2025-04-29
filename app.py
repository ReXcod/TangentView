import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title('Real-World Surface Modeling (Synthetic Example)')

# Grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Derivatives
h = x[1] - x[0]
Zx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * h)
Zy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * h)

# Pick points
points = [(20, 20), (50, 50), (75, 30)]
tangent_planes = []

for i, j in points:
    x0, y0, z0 = X[i, j], Y[i, j], Z[i, j]
    zx, zy = Zx[i, j], Zy[i, j]
    normal = np.array([zx, zy, -1])
    normal /= np.linalg.norm(normal)
    tangent_planes.append((x0, y0, z0, zx, zy, normal))

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

for (x0, y0, z0, zx, zy, normal) in tangent_planes:
    ax.quiver(x0, y0, z0, normal[0], normal[1], normal[2], length=0.5, color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface with Normals')

st.pyplot(fig)
