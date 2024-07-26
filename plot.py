import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from numba import vectorize, float64, jit, njit
import math


a = 1.1821
b = 1.4627
c = 0.1528

@vectorize([float64(float64, float64, float64)])
def get_radius(z,  crown_base, crown_length):

    height = crown_base + crown_length

    z = (z-crown_base) / height

    if z <= 0.:
        return 0.
    if z >= 1.:
        return 0.

    r =  crown_length * c * z**(a - 1.) * (1. - z)**(b - 1)
    return r

# Define the number of points in the z direction
z_points = 50

# Define the number of angular points for the radial symmetry
theta_points = 75

# Generate the z values
z = np.linspace(0, 20., z_points)

# Generate the theta values
theta = np.linspace(0, 2 * np.pi, theta_points)

# Create a meshgrid for theta and z
theta, z = np.meshgrid(theta, z)

# Calculate the radius at each z
radius = get_radius(z, 5., 10.)


# Calculate the x and y coordinates for each point
x = radius * np.cos(theta)
y = radius * np.sin(theta)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each radially symmetric object with an offset in the x direction
num_objects = 5  # Number of objects to plot
offset = 5  # Offset between objects

for i in range(num_objects):
    ax.plot_surface(x + i * offset, y, z, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Radially Symmetric Objects')
ax.set_aspect('equal')

plt.show()