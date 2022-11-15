from data.routes import ThetaFinder, Track, Spline, IntersectionLayout
from mpc.sim import Simulator, SimParams, LinearSystem, BicycleModel
import numpy as np
import matplotlib.pyplot as plt

params = SimParams()
params.N = 9
params.tf = 100
params.ql = 100
params.qc = 0# 0.01
params.Rv = 0.02
params.q_theta = 10
params.Ru = np.zeros((2, 2))
params.Ru[0, 0] = 5
params.Ru[1, 1] = 10
params.d_safe = 0.5
dt = 0.1
lf = 1
lr = 0.5
num_lane = 2
lane_width = 10
track_len = 150

byc_model = BicycleModel(dt, lf, lr)

sys = LinearSystem(byc_model)

track = Track(lane_width, num_lane)
sp = Spline(track)
theta_finder = ThetaFinder(track, sp)

sim = Simulator(params, sys, theta_finder)

x_init_north = np.array([36, 70, -np.pi /2, 0.75]).reshape(-1, 1)
x_init_south = np.array([34, -20, np.pi /2, 0.75]).reshape(-1, 1)
x_init_east = np.array([70, 25, np.pi, 0.75]).reshape(-1, 1)
x_init_west = np.array([0, 35, 0, 0.75]).reshape(-1, 1)
x_init_west_2 = np.array([-10, 23, 0, 0.75]).reshape(-1, 1)

x_init = [x_init_south]
# x_init = [x_init_south, x_init_east,  x_init_north, x_init_west]


sim.set_vehicle_initial_conditions(x_init)

time = np.arange(0, params.tf, dt)

X, Y, al_traj, X_pred, Y_pred = sim.run()


# plot traj: spline of tracks
# traj = al_traj[0]
# x = np.arange(120, 40, -0.1)
# xx = traj.lut_x(x)
# yy = traj.lut_y(x)
# # plt.plot(xx, yy, label = "traj")
#
# plt.plot(X, Y, 'g')
# plt.plot(X_pred, Y_pred, 'r--')
# plt.legend()
# plt.show()
