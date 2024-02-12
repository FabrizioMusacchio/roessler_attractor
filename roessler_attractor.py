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

# time step and total number of steps (affects the duration of the calculation and the gif):
dt = 0.03
step_count = 5000

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
# %% PLOTTING THE ATTRACTOR
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Rössler attractor end state")
plt.tight_layout()
plt.savefig("roessler_attractor.png", dpi=300)
plt.show()
# %% PLOTTING THE ANIMATION
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# adding a text object for time display:
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

def init():
    ax.set_xlim((np.min(path[:,0]), np.max(path[:,0])))
    ax.set_ylim((np.min(path[:,1]), np.max(path[:,1])))
    ax.set_zlim((np.min(path[:,2]), np.max(path[:,2])))
    time_text.set_text('')
    return fig,

def update(num, path, line, time_text):
    line.set_data(path[:num, 0], path[:num, 1])
    line.set_3d_properties(path[:num, 2])
    # Update time display with current simulation time
    time_text.set_text(f'Time = {num*dt:.2f} s')
    print(f"Current step: {num}, Time: {num*dt:.2f} s")
    return line, time_text

line, = ax.plot([], [], [], lw=1)
ani = animation.FuncAnimation(fig, update, step_count, fargs=(path, line, time_text),
                              init_func=init, blit=False, interval=1)

ani.save('roessler_attractor_with_time.gif', writer='imagemagick', fps=30)
plt.show()
print("Done!")
# %% END
