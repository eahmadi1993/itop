from data.routes import ThetaFinder, Track, Spline
from mpc.sim import Simulator, SimParams, LinearSystem, BicycleModel, Optimization
import numpy as np

params = SimParams()
dt = 0.1
lf = 1
lr = 1
byc_model = BicycleModel(dt, lf, lr)

sys = LinearSystem(byc_model)

track = Track(10, 2)
sp = Spline(track)
theta_finder = ThetaFinder(track, sp)

sim = Simulator(params, sys, theta_finder)

x_init = np.array([1, -20, 0, 0]).reshape(-1, 1)

sim.set_vehicle_initial_conditions([x_init])

sim.run()
