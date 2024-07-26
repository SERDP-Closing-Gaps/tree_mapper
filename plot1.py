import numpy as np
import plotly.graph_objects as go

# Define the radius function r(z)
def r(z):
    return np.sin(z) + 2  # Example function

# Define the number of points in the z direction
z_points = 100

# Define the number of angular points for the radial symmetry
theta_points = 100

# Generate the z values
z = np.linspace(0, 2 * np.pi, z_points)

# Generate the theta values
theta = np.linspace(0, 2 * np.pi, theta_points)

# Create a meshgrid for theta and z
theta, z = np.meshgrid(theta, z)

# Calculate the radius at each z
radius = r(z)

# Calculate the x and y coordinates for each point
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Create the plot
fig = go.Figure()

# Plot each radially symmetric object with an offset in the x direction
num_objects = 5  # Number of objects to plot
offset = 5  # Offset between objects

for i in range(num_objects):
    fig.add_trace(go.Surface(x=x + i * offset, y=y, z=z, opacity=0.5))

# Set labels and title
fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    title='Radially Symmetric Objects',
                    aspectmode='cube')

fig.show()
