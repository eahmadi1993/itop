import numpy as np
from numpy import genfromtxt


class Track:
    def __init__(self):
        self.track = {}

    def load_track(self):
        west_east = genfromtxt('we.csv', delimiter= ',')
        west_north = genfromtxt('wn.csv', delimiter= ',')
        west_south = genfromtxt('ws.csv', delimiter= ',')

        south_north = genfromtxt('sn.csv', delimiter= ',')
        south_east = genfromtxt('se.csv', delimiter= ',')
        south_west = genfromtxt('sw.csv', delimiter= ',')

        north_west = genfromtxt('nw.csv', delimiter= ',')
        north_east = genfromtxt('ne.csv', delimiter= ',')
        north_south = genfromtxt('ns.csv', delimiter= ',')

        east_west = genfromtxt('ew.csv', delimiter= ',')
        east_north = genfromtxt('en.csv', delimiter= ',')
        east_south = genfromtxt('es.csv', delimiter= ',')

        self.track["we"] = west_east
        self.track["wn"] = west_north
        self.track["ws"] = west_south

        self.track["sn"] = south_north
        self.track["se"] = south_east
        self.track["sw"] = south_west

        self.track["nw"] = north_west
        self.track["ne"] = north_east
        self.track["ns"] = north_south

        self.track["ew"] = east_west
        self.track["en"] = east_north
        self.track["es"] = east_south


class Spline:
    def __init__(self,track):
        self.track = track

    def normalizedSplineInterp(self):
        pass

    def splineInterp(self):
        t = np.arange(0,len())






