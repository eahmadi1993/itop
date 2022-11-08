import random
import numpy as np
from numpy import genfromtxt
import casadi as cs
import scipy.integrate as integrate


# This class defines 12 possible movements inside intersection.
class Track:
    def __init__(self, lane_width, num_lane):
        self.track_data = {}
        self.track_width = lane_width * num_lane

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
        east_north = genfromtxt('en_new.csv', delimiter=',')
        east_south = genfromtxt('es_new.csv', delimiter=',')
        # east_north = genfromtxt('en.csv', delimiter = ',')
        # east_south = genfromtxt('es.csv', delimiter = ',')

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


# This class is used in class spline, only to save all its attributes in traj
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
            # if tr_name == "es":
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

        # here, tarj is an object of class Trajectory for giving values to attributes of class Trajectory
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
        umin = np.arange(len(xt) - 1)  # 0:229
        umax = np.arange(1, len(xt))  # 1:231
        l = np.zeros((len(umin), 1))

        def haux(u):
            out = np.sqrt(d_lut_x(u, 0) ** 2 + d_lut_y(u, 0) ** 2)
            return out.elements()[0]

        for i in range(len(umin)):
            l[i] = integrate.quad(haux, float(umin[i]), float(umax[i]))[0]
        return l


class ThetaFinder:
    def __init__(self, track: Track, spline: Spline):
        self.track = track
        self.spline = spline

        self.mytrack = None
        self.mytraj = None

        self.init_x = None
        self.init_y = None
        self.paths = {
            "n": ["ns", "ne", "nw"],
            "s": ["sw", "se", "sn"],
            "e": ["es", "en", "ew"],
            "w": ["we", "ws", "wn"]
        }

    def set_initial_conditions(self, init_x, init_y):
        self.init_x = init_x
        self.init_y = init_y
        self.mytrack, self.mytraj = self.find_track_traj()

    def find_path(self):
        """ this function determines the direction each vehicle enter the plaza, south(s), north(n), east(e), and west(w)
        w, s, e, and n are the keys of dictionary paths and each of them has three values
        for example "s" has sn, se, and sw"""
        if self.init_x < 32.1 - 10:
            return "w"
        if self.init_y < 29.2 - 10:
            return "s"
        if self.init_x > 32.1 + 10:
            return "e"
        if self.init_y > 29.2 + 10:
            return "n"

    def find_track_traj(self):
        path = self.find_path()
        # tr_name = random.choice(self.paths[path])
        tr_name = "ws"
        trc_data = self.track.track_data[tr_name]
        trj_data = self.spline.my_traj[tr_name]
        return trc_data, trj_data

    def find_theta(self, posx, posy):
        track, traj = self.mytrack, self.mytraj

        trc_x = track[0, :]
        trc_y = track[1, :]
        n_track = len(trc_x)
        distance_x = trc_x - posx * np.ones((1, n_track))
        distance_y = trc_y - posy * np.ones((1, n_track))

        squared_dist = distance_x[0] ** 2 + distance_y[0] ** 2
        min_index = np.argmin(squared_dist)
        e = squared_dist[min_index]

        if min_index == 0:
            next_index = 1
            prev_index = n_track
        elif min_index == n_track:
            next_index = 0
            prev_index = n_track - 1
        else:
            next_index = min_index + 1
            prev_index = min_index - 1

        closest_index = min_index

        cosine = np.dot(
            np.array([posx, posy]).reshape(1, -1) - track[:, min_index].reshape(1, -1),
            track[:, prev_index].reshape(-1, 1) - track[:, min_index].reshape(-1, 1)
        )

        if cosine > 0:
            min_index2 = prev_index
        else:
            min_index2 = min_index
            min_index = next_index

        if e != 0:
            cosine = np.dot(
                np.array([posx, posy]).reshape(1, -1) - track[:, min_index2].reshape(1, -1),
                track[:, min_index].reshape(-1, 1) - track[:, min_index2].reshape(-1, 1)) / (
                             np.linalg.norm(
                                 np.array([posx, posy]).reshape(-1, 1) - track[:, min_index2].reshape(-1, 1)) *
                             np.linalg.norm(track[:, min_index].reshape(-1, 1) - track[:, min_index2].reshape(-1, 1))
                     )
        else:
            cosine = 0

        traj_cl = np.concatenate((np.zeros(1), traj.cl.reshape(-1, )), axis = 0)
        theta = traj_cl[min_index2]
        theta = theta + cosine * np.linalg.norm(
            np.array([posx, posy]).reshape(-1, 1) - track[:, min_index2].reshape(-1, 1), 2)
        return float(theta)
