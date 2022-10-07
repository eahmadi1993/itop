from data.routes import Track, Spline, ThetaFinder
num_lane = 2
lane_width = 10
tr = Track(lane_width, num_lane)
sp = Spline(tr)

tf = ThetaFinder(tr, sp)
tf.set_initial_conditions(-12.3125, 35.3)
print(tf.find_theta(-12.3125, 35.3))

# trc_len = []
# for key, value in sp.my_traj.items():
#     trc_len.append(float(value.cl[-1]))
# print(max(trc_len))