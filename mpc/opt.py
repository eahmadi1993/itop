import casadi as cs
import numpy as np

from mpc.sim import Simulator


class Optimization:
    def __init__(self, sim: Simulator):
        self.opti = cs.Opti()
        self.sim = sim
        self.states = []
        self.controls = []
        self.s = []
        self.lam = []
        self.x_app = None
        self.theta_app = None

    def decision_vars(self):
        # ? I don't know for k vehicles, how many s do we have?
        # self.s = self.opti.variable(self.sim.sys.m, self.sim.params.N)
        for i in range(self.sim.num_veh):
            self.states = self.opti.variable(self.sim.sys.n, self.sim.params.N)
            self.controls = self.opti.variable(self.sim.sys.m, self.sim.params.N - 1)
            for j in range(self.sim.num_veh):
                if i != j:
                    self.lam = self.opti.variable(self.sim.sys.n, self.sim.params.N)

    def set_app_points(self, x_app, theta_app):
        self.x_app = x_app
        self.theta_app = theta_app

    def angle_of_theta(self):
        pass

    def compute_gamma(self):
        pass

    def app_contouring_error(self):
        pass

    def app_lag_error(self):
        pass

    def obj_func(self):
        total_obj = 0
        _, traj = self.sim.theta_finder.find_track_traj()
        for k in range(self.sim.num_veh):
            for i in range(self.sim.params.N):
                phi = np.arctan2(traj[k].d_lut_y(self.theta_app[k][i], 0), traj[k].d_lut_x(self.theta_app[k][i], 0))

    def constraints(self):
        pass

    def run_opt(self):
        pass
