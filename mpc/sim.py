from data.routes import ThetaFinder
import numpy as np


class BicycleModel:
    def __init__(self, dt, lf, lr):
        self.lf = lf
        self.lr = lr
        self.dt = dt
        self.xbar = None
        self.ubar = None

    def set_lin_points(self, xbar, ubar):
        self.xbar = xbar
        self.ubar = ubar

    def _compute_beta_gamma(self):
        alpha = self.lr / (self.lf + self.lr)
        beta = np.arctan(alpha * np.tan(self.ubar[1]))
        gamma = alpha / (np.cos(self.ubar[1]) ** 2 * (1 + alpha ** 2 * np.tan(self.ubar[1]) ** 2))
        return beta, gamma

    def _compute_xbar_dot(self):
        beta, _ = self._compute_beta_gamma()
        xbar_dot = [[float(self.xbar[3] * np.cos(self.xbar[2] + beta))],
                    [float(self.xbar[3] * np.sin(self.xbar[2] + beta))],
                    [float(self.xbar[3] * np.sin(beta) / self.lr)],
                    [float(self.ubar[0])]]
        xbar_dot = np.array(xbar_dot, dtype = object)
        return xbar_dot

    def f_a(self):
        beta, _ = self._compute_beta_gamma()
        a_mat = [[0, 0, -self.xbar[3] * np.sin(self.xbar[2] + beta), np.cos(self.xbar[2] + beta)],
                 [0, 0, self.xbar[3] * np.cos(self.xbar[2] + beta), np.sin(self.xbar[2] + beta)],
                 [0, 0, 0, np.sin(beta) / self.lr],
                 [0, 0, 0, 0]]
        a_mat = np.array(a_mat, dtype = object)
        return a_mat

    def f_b(self):
        beta, gamma = self._compute_beta_gamma()
        b_mat = [[0, -gamma * self.xbar[3] * np.sin(self.xbar[2] + beta)],
                 [0, gamma * self.xbar[3] * np.cos(self.xbar[2] + beta)],
                 [0, gamma * np.cos(beta)],
                 [1, 0]]
        b_mat = np.array(b_mat, dtype = object)
        return b_mat

    def f_d(self):
        xbar_dot = self._compute_xbar_dot()
        a_mat = self.f_a()
        b_mat = self.f_b()
        d_vec = xbar_dot - a_mat @ self.xbar - b_mat @ self.ubar
        return d_vec


class LinearSystem:

    def __init__(self, bicycle_model: BicycleModel):
        self.dt = bicycle_model.dt
        self.model = bicycle_model

    def update_states(self, x, u, xbar, ubar):
        a_mat, b_mat, d_vec = self.linearize_at(xbar, ubar)
        x = x + self.dt * (a_mat @ x + b_mat @ u + d_vec)
        x = np.array([float(item) for item in x]).reshape(-1, 1)
        return x

    def linearize_at(self, xbar, ubar):
        self.model.set_lin_points(xbar, ubar)
        a_mat = self.model.f_a()
        b_mat = self.model.f_b()
        d_vec = self.model.f_d()
        return a_mat, b_mat, d_vec


class SimParams:
    def __init__(self):
        self.N = None  # prediction horizon
        self.tf = None  # final time
        self.num_veh = None  # number of vehicles
        self.d_safe = None


class Simulator:
    def __init__(self, params: SimParams, sys: LinearSystem, theta_finder: ThetaFinder):
        self.params = params
        self.sys = sys
        self.theta_finder = theta_finder

    def optimize(self):
        pass

    def get_states(self):
        pass

    def run(self):
        pass

    def get_results(self):
        pass
