import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos, sqrt

# Some Constants
G = 6.673e-11
M = 5.972e24

RADIUS = 6378100
ISS_HEIGHT = 450e3
ISS_VELOCITY = 7500
ISS_TIME = 90*60

NORTH = -45 * (pi / 180)
EAST = -60 * (pi / 180)

# +z is up
startpos = np.array([cos(NORTH) * cos(EAST), cos(NORTH) * sin(EAST), sin(NORTH)])


def solve_quadratic(a, b, c):
    D = b**2 - 4 * a * c
    s1 = (-b + sqrt(D)) / (2 * a) # solution 1
    s2 = (-b - sqrt(D)) / (2 * a) # solution 2
    return s1, s2


def get_orbit_norm(s):
    sx, sy, sz = s

    phi = 51.6 * (pi / 180)

    p1 = -sy / sx
    p2 = -cos(phi) * sz / sx

    a = p1**2 + 1
    b = 2 * p1 * p2
    c = p2**2 - sin(phi)**2

    y1, _ = solve_quadratic(a, b, c)
    
    x1 = p1 * y1 + p2
    # x2 = p1 * y2 + p2

    z = cos(phi)

    return np.array([x1, y1, z])

rotateZ = lambda a: np.array([[cos(a), -sin(a), 0],
                             [sin(a), cos(a), 0],
                             [0, 0, 1]])

orbit_norm = get_orbit_norm(startpos)
tau = np.cross(orbit_norm, startpos)

r0 = startpos * (RADIUS + ISS_HEIGHT)
v0 = tau * ISS_VELOCITY

def ode45(odefunc, t, x0):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        k1 = odefunc(t[i - 1], x[i - 1])
        k2 = odefunc(t[i - 1] + dt / 2, x[i - 1] + dt / 2 * k1)
        k3 = odefunc(t[i - 1] + dt / 2, x[i - 1] + dt / 2 * k2)
        k4 = odefunc(t[i - 1] + dt, x[i - 1] + dt * k3)
        x[i] = x[i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, x

odefunc = lambda t, x: np.concatenate((x[3:6], -G * M * x[:3] / np.linalg.norm(x[:3]) ** 3))
tspan = np.linspace(0, 1 * ISS_TIME, int(10e5))
x0 = np.concatenate((r0, v0))

t, x = ode45(odefunc, tspan, x0)
trajectory = x[:, :3]

trajectory_corrected = np.zeros(trajectory.shape)
for i in range(len(t)):
    ti = t[i]
    angle_rotation = -2*pi * ti / (24 * 60 * 60)
    point = trajectory[i, :]
    point_corrected = rotateZ(angle_rotation) @ point
    trajectory_corrected[i, :] = point_corrected

N = 100
phi = np.linspace(0, 2*pi, N)
theta = np.linspace(0, pi, N)
theta, pi = np.meshgrid(theta, phi)

_x = RADIUS * np.cos(phi) * np.sin(theta)
y = RADIUS * np.sin(phi) * np.sin(theta)
z = RADIUS * np.cos(theta)

fig = plt.figure(figsize=[10,10])
ax = plt.axes(projection='3d')

ax.plot_surface(_x, y, z, color='green', alpha=.5)

tc = trajectory_corrected
ax.plot3D(tc[:, 0], tc[:, 1], tc[:, 2], linewidth=4)

ax.set_title('Satellite orbit')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig = plt.figure(figsize=(20,10), dpi=80)

ax = fig.add_subplot(211)
ax.plot(tspan, x[:, 0:3])
ax.set_title('Coordinates')

ax1 = fig.add_subplot(212)
ax1.plot(tspan, x[:, 3:6])
ax.set_title('Velocity')

plt.show()
plt.savefig('plot.png')