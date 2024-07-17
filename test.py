import numpy as np
from numba import jit
import numpy.random as random
from skimage.feature import peak_local_max
import math 
from scipy.special import beta
import matplotlib.pyplot as plt
from scipy.special import expit


"""
Randomly generates tree locations given a crown height map and a radius 
field that defines the minimum spacing between trees. 
"""

r = 2.5
height0 = 10.
height1 = 11.
dh = abs(height0 - height1)
xs = np.linspace(-10., 10.)     
ys = expit((xs - r) / 2.*dh)


plt.plot(xs, ys)
plt.show()
