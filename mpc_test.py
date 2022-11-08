from data.routes import ThetaFinder, Track, Spline
from mpc.sim import Simulator, SimParams, LinearSystem, BicycleModel, Optimization, NonlinearSystem
import numpy as np
import matplotlib.pyplot as plt

params = SimParams()
params.N = 40
params.tf = 10
params.ql = 100
params.qc = 10
params.Rv = 0.002
params.q_theta = 0.1
params.Ru = np.zeros((2, 2))
params.Ru[0, 0] = 5
params.Ru[1, 1] = 0.5
dt = 0.1
lf = 1.005
lr = 1.538
byc_model = BicycleModel(dt, lf, lr)

sys = LinearSystem(byc_model)

track = Track(10, 2)
sp = Spline(track)
theta_finder = ThetaFinder(track, sp)

sim = Simulator(params, sys, theta_finder)

# x_init = np.array([27, 70, -np.pi /2, 0.1]).reshape(-1, 1)  # for North vehicles
# x_init = np.array([36, -10, np.pi /2, 0.5]).reshape(-1, 1)  # for South vehicles
x_init = np.array([90, 25, np.pi, 0.5]).reshape(-1, 1)       # NOT worked! for East vehicles
# x_init = np.array([-10, 25, 0, 0]).reshape(-1, 1)        # NOT worked! for West vehicles

sim.set_vehicle_initial_conditions([x_init])

time = np.arange(0, params.tf, dt)

X, Y, al_traj, X_pred, Y_pred = sim.run()

traj = al_traj[0]

x = np.arange(120, 40, -0.1)
xx = traj.lut_x(x)
yy = traj.lut_y(x)

plt.plot(xx, yy, label = "traj")
plt.plot(X, Y)
# plt.plot(X_pred, Y_pred, 'r--')
plt.legend()
plt.show()
