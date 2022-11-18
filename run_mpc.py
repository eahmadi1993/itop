from routes import ThetaFinder, Track, Spline, IntersectionLayout
from sim_mpc import MPCC, MPCCParams
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from system import BicycleModel, LinearSystem
import matplotlib.animation as animation


sim = MPCC(params, sys, theta_finder)


sim.set_vehicle_initial_conditions(x_init)

X, Y, al_traj, X_pred, Y_pred, speed = sim.run()

intersection = IntersectionLayout(track, lane_width, track_len)
intersection.plot_intersection(anim = True)

colors = ["b", "r", "g", "c", "m", "k"]


def animate(i):
    for j in range(len(x_init)):
        plt.subplot(211)
        plt.plot(X[j][0:i], Y[j][0:i], colors[j])
        plt.subplot(212)
        plt.plot(time[0:i], speed[j][0:i], colors[j])


print("generating video ... ")
anim = FuncAnimation(intersection.fig, animate, frames = len(time), interval = 75, repeat = False)
writervideo = animation.FFMpegWriter(fps = 60)
anim.save(r'video.mp4', writer = writervideo, dpi = 250)
print("generating video done. ")

