from data.routes import ThetaFinder, Track, Spline
from mpc.sim import Simulator, SimParams, LinearSystem, BicycleModel, Optimization, NonlinearSystem
import numpy as np
import matplotlib.pyplot as plt

params = SimParams()
params.N = 20
params.tf = 20
dt = 0.05
lf = 1.105
lr = 1.738
byc_model = BicycleModel(dt, lf, lr)

sys = LinearSystem(byc_model)

track = Track(10, 2)
sp = Spline(track)
theta_finder = ThetaFinder(track, sp)

sim = Simulator(params, sys, theta_finder)

x_init = np.array([37, -10, 0, 0.05]).reshape(-1, 1)

sim.set_vehicle_initial_conditions([x_init])

time = np.arange(0, params.tf, dt)

sim.run()

# plt.plot(X,Y)
# plt.show()
