
import numpy as np
import matplotlib.pyplot as plt



a = 1.1821
b = 1.4627
c = 0.1528


zs = np.linspace(0.,25.,50)


def intersection_volume(z, d, cl0, cl1, cb0, cb1):
    r0 = get_radius(z, cl0, cb0)
    r1 = get_radius(z, cl1, cb1)

    x = 4.*d**2 *r0**2 - (d**2 - r1**2 + r0**2)**2
    x[x < 0.] = 0.
    dz = abs(z[1] - z[0])

    a = (1./d) * np.sqrt(x)
    V = (a*dz).sum()

    return V 


def get_radius(z, crown_length, crown_base):

    r = np.zeros_like(zs)
    height = crown_base + crown_length

    z = (z-crown_base) / height
    z[z < 0.] = 0.
    z[z > 1.] = 1.

    r =  crown_length * c * z**(a - 1.) * (1. - z)**(b - 1)
    return r


V = intersection_volume(zs, 1., 10., 5., 10., 5.)
print(V)

quit()
plt.subplot(2,1,1)
plt.plot(zs, a)

plt.subplot(2,1,2)
plt.plot(zs, r0)
plt.plot(zs, -r0)
plt.plot(zs, d-r1)
plt.plot(zs, d+r1)
plt.show()