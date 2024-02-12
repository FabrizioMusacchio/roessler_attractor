"""
Just a script to generate a gif of the Rössler attractor.

author: Fabrizio Musacchio
date: Feb 12, 2024

For reproducibility:

conda create -n attractor_networks -y python=3.11
conda activate attractor_networks
conda install mamba -y
mamba install -y numpy matplotlib scikit-learn ipykernel

"""
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# %% PARAMETERS
# parameters for the Rössler attractor:
a, b, c = 0.2, 0.2, 5.7

# time step and total number of steps:
dt = 0.01
step_count = 10000

# initial condition:
x0, y0, z0 = 0.0, 0.0, 0.0
# %% FUNCTIONS
# function to calculate the derivatives:
def roessler_attractor(state, a, b, c):
    x, y, z = state
    dx = -y - z
    dy = x + a*y
    dz = b + z*(x - c)
    return np.array([dx, dy, dz])

# RK4 step function:
def rk4_step(state, dt, a, b, c):
    k1 = roessler_attractor(state, a, b, c)
    k2 = roessler_attractor(state + dt/2 * k1, a, b, c)
    k3 = roessler_attractor(state + dt/2 * k2, a, b, c)
    k4 = roessler_attractor(state + dt * k3, a, b, c)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
# %% MAIN
# generating the trajectory:
path = np.zeros((step_count, 3))
path[0] = [x0, y0, z0]
for i in range(1, step_count):
    path[i] = rk4_step(path[i-1], dt, a, b, c)
# %% PLOTTING/ANIMATION
# animation:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.set_xlim((np.min(path[:,0]), np.max(path[:,0])))
    ax.set_ylim((np.min(path[:,1]), np.max(path[:,1])))
    ax.set_zlim((np.min(path[:,2]), np.max(path[:,2])))
    return fig,

def update(num, path, line):
    line.set_data(path[:num, 0], path[:num, 1])
    line.set_3d_properties(path[:num, 2])
    return line,

line, = ax.plot([], [], [], lw=2)
ani = animation.FuncAnimation(fig, update, step_count, fargs=(path, line),
                              init_func=init, blit=False)

ani.save('roessler_attractor.gif', writer='imagemagick', fps=30)
plt.show()
# %% END
