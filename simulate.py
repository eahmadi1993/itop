import numpy as np
from cam import Simulator
from routes import Track, Spline, ThetaFinder
from sim_mpc import MPCCParams, MPCC
from system import BicycleModel, LinearSystem

params = MPCCParams()
params.N = 20
params.tf = 15
params.ql = 10
params.qc = 0.01  # 0.01
params.Rv = 0.02
params.q_theta = 0.10
params.Ru = np.zeros((2, 2))
params.Ru[0, 0] = 5
params.Ru[1, 1] = 10
params.d_safe = 1
dt = 0.1
lf = 0.5
lr = 1
num_lane = 2
lane_width = 10
track_len = 150
time = np.arange(0, params.tf, dt)

byc_model = BicycleModel(dt, lf, lr)
sys = LinearSystem(byc_model)
track = Track(lane_width, num_lane)
sp = Spline(track)
theta_finder = ThetaFinder(track, sp)
mpcc = MPCC(params, sys, theta_finder)
simulator = Simulator(mpcc)
simulator.run_simulation()
