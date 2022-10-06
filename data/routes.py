import numpy as np
from numpy import genfromtxt
import casadi as cs
import scipy.integrate as integrate


class Track:
    def __init__(self):
        self.track_data = {}

    def load_track(self):
        west_east = genfromtxt('we.csv', delimiter = ',')
        west_north = genfromtxt('wn.csv', delimiter = ',')
        west_south = genfromtxt('ws.csv', delimiter = ',')

        south_north = genfromtxt('sn.csv', delimiter = ',')
        south_east = genfromtxt('se.csv', delimiter = ',')
        south_west = genfromtxt('sw.csv', delimiter = ',')

        north_west = genfromtxt('nw.csv', delimiter = ',')
        north_east = genfromtxt('ne.csv', delimiter = ',')
        north_south = genfromtxt('ns.csv', delimiter = ',')

        east_west = genfromtxt('ew.csv', delimiter = ',')
        east_north = genfromtxt('en.csv', delimiter = ',')
        east_south = genfromtxt('es.csv', delimiter = ',')

        self.track_data["we"] = west_east
        self.track_data["wn"] = west_north
        self.track_data["ws"] = west_south

        self.track_data["sn"] = south_north
        self.track_data["se"] = south_east
        self.track_data["sw"] = south_west

        self.track_data["nw"] = north_west
        self.track_data["ne"] = north_east
        self.track_data["ns"] = north_south

        self.track_data["ew"] = east_west
        self.track_data["en"] = east_north
        self.track_data["es"] = east_south


class Trajectory:
    def __init__(self):
        self.lut_x = None
        self.lut_y = None
        self.cl = None
        self.d_lut_x = None
        self.d_lut_y = None
        self.dd_lut_x = None
        self.dd_lut_y = None


class Spline:
    def __init__(self, track: Track):
        self.track = track
        self.my_traj = {}
        self.track.load_track()
        self.get_traj()

    def get_traj(self):
        for tr_name, tr_data in self.track.track_data.items():
            traj = self.splinify(tr_data)
            self.my_traj[tr_name] = traj
        return self.my_traj

    def splinify(self, tr_data):
        xt = tr_data[0, :]
        yt = tr_data[1, :]
        lut_x, lut_y, cl = self.normalized_spline_interp(xt, yt)
        d_lut_x = lut_x.jacobian()
        d_lut_y = lut_y.jacobian()
        dd_lut_x = d_lut_x.jacobian()
        dd_lut_y = d_lut_y.jacobian()

        traj = Trajectory()
        traj.lut_x = lut_x
        traj.lut_y = lut_y
        traj.d_lut_x = d_lut_x
        traj.d_lut_y = d_lut_y
        traj.dd_lut_x = dd_lut_x
        traj.dd_lut_y = dd_lut_y
        traj.cl = cl
        return traj

    def normalized_spline_interp(self, xt, yt):
        lut_x, lut_y = self.spline_interp(xt, yt)
        l = self.spline_length(lut_x, lut_y, xt, yt)
        cl = np.cumsum(l)
        cl_concat = np.concatenate((np.zeros((1)), cl), axis = 0)
        lut_x = cs.interpolant("LUT", "bspline", [cl_concat], xt)
        lut_y = cs.interpolant("LUT", "bspline", [cl_concat], yt)
        return lut_x, lut_y, cl.reshape(-1, 1)

    def spline_interp(self, xt, yt):
        t = np.arange(0, len(xt), 1)
        lut_x = cs.interpolant("LUT", "bspline", [t], xt)
        lut_y = cs.interpolant("LUT", "bspline", [t], yt)
        return lut_x, lut_y

    def spline_length(self, lut_x, lut_y, xt, yt):
        d_lut_x = lut_x.jacobian()
        d_lut_y = lut_y.jacobian()
        umin = np.arange(len(xt) - 1) #0:229
        umax = np.arange(1, len(xt))  #1:231
        l = np.zeros((len(umin), 1))

        def haux(u):
            out = np.sqrt(d_lut_x(u, 0) ** 2 + d_lut_y(u, 0) ** 2)
            return out.elements()[0]

        for i in range(len(umin)):
            l[i] = integrate.quad(haux, float(umin[i]), float(umax[i]))[0]
        return l
