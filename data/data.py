from numpy import genfromtxt


class Track:
    def __init__(self):
        self.track = {}

    def load_track(self):
        westEast = genfromtxt('we.csv', delimiter=',')
        westNorth = genfromtxt('wn.csv', delimiter=',')
        westSouth = genfromtxt('ws.csv', delimiter=',')

        southNorth = genfromtxt('sn.csv', delimiter=',')
        southEast = genfromtxt('se.csv', delimiter=',')
        southWest = genfromtxt('sw.csv', delimiter=',')

        northWest = genfromtxt('nw.csv', delimiter=',')
        northEast = genfromtxt('ne.csv', delimiter=',')
        northSouth = genfromtxt('ns.csv', delimiter=',')

        eastWest = genfromtxt('ew.csv', delimiter=',')
        eastNorth = genfromtxt('en.csv', delimiter=',')
        eastSouth = genfromtxt('es.csv', delimiter=',')

        self.track["we"] = westEast
        self.track["wn"] = westNorth
        self.track["ws"] = westSouth

        self.track["sn"] = southNorth
        self.track["se"] = southEast
        self.track["sw"] = southWest

        self.track["nw"] = northWest
        self.track["ne"] = northEast
        self.track["ns"] = northSouth

        self.track["ew"] = eastWest
        self.track["en"] = eastNorth
        self.track["es"] = eastSouth


class Spline:
    pass