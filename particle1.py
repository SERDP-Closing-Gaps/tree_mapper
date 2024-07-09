# Authored by Tiantian Liu, Taichi Graphics.
import math

import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt


ti.init(arch=ti.gpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1

# number of planets
N = 5000
# unit mass
m = 1
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = 2
# init vel
init_vel = 0.

# time-step size
h = 1e-3
# substepping
substepping = 10

# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)


xs = np.linspace(0.,1.,100)
ys = np.linspace(0.,1.,100)
xx, yy = np.meshgrid(xs, ys)

phi = -np.exp( (-(xx - 0.5)**2 - (yy - 0.5)**2) / (0.5**2) )
phi_x, phi_y = np.gradient(-phi, 1.)
phi_x *= 100.
phi_y *= 100.



# Convert the NumPy array to a Taichi field
phi_x_field = ti.field(dtype=ti.f32, shape=phi_x.shape)
phi_x_field.from_numpy(phi_x)

phi_y_field = ti.field(dtype=ti.f32, shape=phi_y.shape)
phi_y_field.from_numpy(phi_y)


@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    for i in range(N):
        theta = ti.random() * 2 * math.pi
        r = (ti.sqrt(ti.random()) * 0.6 + 0.4) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center[None] + offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel


@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = [0.0, 0.0]
        #p = pos[i]

    # compute gravitational force
    for i in range(N):
        p = pos[i]

        i0 = int(p[0] * 100.) 
        i1 = int(p[1] * 100.)

        f0 = phi_x_field[i0, i1]
        f1 = phi_y_field[i0, i1]

        v = vel[i]
        v_mag = v.norm(1e-6)
        drag = 100. * v_mag**2 * v
        
        force[i] = [f0*1e2, f1*1e2] - drag

        for j in range(N):
            if i != j:  
                diff = p - pos[j]
                r = diff.norm(1e-6)
                if r < 0.25:
                    force[i] += (0.05 / r) * (diff / r) 

                


@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        # symplectic euler
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]


def main():
    gui = ti.GUI("N-body problem", (800, 800))

    initialize()
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == "r":
                initialize()
            elif e.key == ti.GUI.SPACE:
                paused[None] = not paused[None]

        if not paused[None]:
            for i in range(substepping):
                compute_force()
                update()

        gui.circles(pos.to_numpy(), color=0xFFFFFF, radius=planet_radius)
        gui.show()


if __name__ == "__main__":
    main()