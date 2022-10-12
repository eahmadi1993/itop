import copy
import numpy as np
from data.routes import ThetaFinder


class SimParams:
    def __init__(self):
        self.N = None    # prediction horizon
        self.tf = None   # final time
        self.d_safe = None
        self.simN = None
        self.Q = None
        self.q = None
        self.R = None


class BicycleModel:
    def __init__(self, dt, lf, lr):
        self.lf = lf      # distance from the center of the mass of the vehicle to the front axles
        self.lr = lr      # distance from the center of the mass of the vehicle to the rear axles,
        self.dt = dt      # discretization time
        self.xbar = None  # linearization points
        self.ubar = None  # linearization points
        self.m = 2        # number of control inputs
        self.n = 4        # number of states

    def set_lin_points(self, xbar, ubar):
        self.xbar = xbar
        self.ubar = ubar

    def _compute_beta_gamma(self):
        """β is the angle of the current velocity of the center of mass
             with respect to the longitudinal axis of the vehicle. """
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
        self.m = self.model.m
        self.n = self.model.n

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


class NonlinearSystem:
    def __init__(self, dt, lr, lf):
        self.dt = dt
        self.lr = lr
        self.lf = lf

    def _compute_beta(self, u):
        """β is the angle of the current velocity of the center of mass
         with respect to the longitudinal axis of the vehicle. """
        alpha = self.lr / (self.lf + self.lr)
        beta = np.arctan(alpha * np.tan(u[1]))
        return beta

    def update_nls_states(self, x, u):
        beta = self._compute_beta(u)
        f = [[float(x[3] * np.cos(x[2] + beta))],
             [float(x[3] * np.sin(x[2] + beta))],
             [float(x[3] * np.sin(beta) / self.lr)],
             [float(u[0])]]
        f = np.array(f)
        x = x + self.dt * f
        return x


class Simulator:
    def __init__(self, params: SimParams, sys: LinearSystem, theta_finder: ThetaFinder):
        self.params = params
        self.sys = sys
        self.theta_finder = theta_finder
        self.num_veh = None
        self.x_init_list = None
        self.theta_init_list = []
        self.u_init_list = []

    def set_vehicle_initial_conditions(self, x_init_list):
        self.num_veh = len(x_init_list)
        self.x_init_list = x_init_list
        m = self.sys.m

        for i in range(self.num_veh):
            self.theta_init_list.append(
                self.theta_finder.find_theta(self.x_init_list[i][0], self.x_init_list[i][1])
            )

            self.u_init_list.append(
                np.zeros((m, 1))
            )

    def optimize(self):
        pass

    def update_vehicles_states(self, x_prev_list, u_opt_list, x_bar_list, u_bar_list):
        updated_x = []
        updated_theta = []
        for i in range(self.num_veh):
            x = self.sys.update_states(x_prev_list[i], u_opt_list[i], x_bar_list[i], u_bar_list[i])
            theta = self.theta_finder.find_theta(x[0], x[1])
            updated_x.append(x)
            updated_theta.append(theta)
        return updated_x, updated_theta

    def run(self):
        time = np.arange(0, self.params.simN, self.sys.dt)
        x = self.x_init_list
        theta = self.theta_init_list
        u = self.u_init_list
        for t_ind, t in enumerate(time):   # MPC loop
            xbar = copy.deepcopy(x)
            ubar = copy.deepcopy(u)
            # optimization is solved here
            x, theta = self.update_vehicles_states(x, u, xbar, ubar)

    def get_results(self):
        pass
