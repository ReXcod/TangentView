import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title('Interactive Surface Modeling with Tangent Planes and Normals')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')

# 1. Select Surface Equation
surface_choice = st.sidebar.selectbox(
    'Choose Surface Equation',
    ['sin(x)*cos(y)', 'x^2 + y^2', 'exp(-(x^2 + y^2))']
)

# 2. Choose Resolution (grid size)
resolution = st.sidebar.slider(
    'Surface Grid Resolution', min_value=50, max_value=200, value=100, step=10
)

# 3. Number of Points to Calculate Tangent Planes for
num_points = st.sidebar.number_input(
    'Number of Points for Tangent Planes', min_value=1, max_value=10, value=3
)

# 4. Enter Points as (x, y) pairs
points = []
for i in range(num_points):
    x_input = st.sidebar.number_input(f"Point {i+1} X", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    y_input = st.sidebar.number_input(f"Point {i+1} Y", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    points.append((x_input, y_input))

# Function to define surfaces
def surface_function(X, Y, choice):
    if choice == 'sin(x)*cos(y)':
        return np.sin(X) * np.cos(Y)
    elif choice == 'x^2 + y^2':
        return X**2 + Y**2
    elif choice == 'exp(-(x^2 + y^2))':
        return np.exp(-(X**2 + Y**2))

# Generate the surface
x = np.linspace(-5, 5, resolution)
y = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x, y)
Z = surface_function(X, Y, surface_choice)

# Compute partial derivatives (approximations)
h = x[1] - x[0]
Zx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2 * h)
Zy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2 * h)

# Calculate tangent planes and normal vectors for each user-selected point
tangent_planes = []
for (x0, y0) in points:
    i = np.abs(x - x0).argmin()  # find the nearest x index
    j = np.abs(y - y0).argmin()  # find the nearest y index
    z0 = Z[i, j]
    zx, zy = Zx[i, j], Zy[i, j]
    
    # Normal vector
    normal = np.array([zx, zy, -1])
    normal /= np.linalg.norm(normal)  # Normalize

    tangent_planes.append((x0, y0, z0, zx, zy, normal))

# Plotting the surface and normal vectors
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plot normal vectors
for (x0, y0, z0, zx, zy, normal) in tangent_planes:
    ax.quiver(x0, y0, z0, normal[0], normal[1], normal[2], length=0.5, color='red')

# Plot tangent planes (small patches around each selected point)
for (x0, y0, z0, zx, zy, normal) in tangent_planes:
    xx = np.linspace(x0 - 0.5, x0 + 0.5, 10)
    yy = np.linspace(y0 - 0.5, y0 + 0.5, 10)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = z0 + zx * (XX - x0) + zy * (YY - y0)
    ax.plot_surface(XX, YY, ZZ, color='red', alpha=0.5)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface with Tangent Planes and Normal Vectors')

# Display the plot
st.pyplot(fig)

# Show the tangent plane equations below the plot
st.subheader('Tangent Plane Equations')
for idx, (x0, y0, z0, zx, zy, normal) in enumerate(tangent_planes):
    equation = f"z = {z0:.2f} + ({zx:.2f})(x - {x0:.2f}) + ({zy:.2f})(y - {y0:.2f})"
    st.write(f"**Point {idx+1}:** {equation}")
