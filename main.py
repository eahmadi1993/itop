from numpy.random import randn

from data.routes import Track, Spline, ThetaFinder
from mpc.sim import LinearSystem, BicycleModel, NonlinearSystem

# test class LinearSystem and BicycleModel of folder sim in package mpc
bcs = BicycleModel(dt = 0.1, lf = 1, lr = 1)
system = LinearSystem(bcs)
x = randn(4, 1)
xbar = randn(4, 1)
u = randn(2, 1)
ubar = randn(2, 1)
print(system.update_states(x, u, xbar, ubar))

# # test class NonLinearSystem of folder sim in package mpc
# nls = NonlinearSystem(dt = 0.1, lf = 1,lr = 1)
# x = randn(4, 1)
# u = randn(2,1)
# print(nls.update_nls_states(x, u))



# test class Spline, Track and FindTheta of folder routes in package data
# num_lane = 2
# lane_width = 10
# tr = Track(lane_width, num_lane)
# tr.load_track()
# print(tr.track_data["sn"][0][0])  #32.1

# sp = Spline(tr)
# tf = ThetaFinder(tr, sp)
# tf.set_initial_conditions(-12.3125, 35.3)
# print(tf.find_theta(-12.3125, 35.3))




#Eli: I wrote it for calculating track lengt of each approach
# trc_len = []
# for key, value in sp.my_traj.items():
#     trc_len.append(float(value.cl[-1]))
# print(max(trc_len))
