import numpy as np
import casadi as cs

class BicycleModel:
    def __init__(self, dt, lf, lr):
        self.lf = lf  # distance from the center of the mass of the vehicle to the front axles
        self.lr = lr  # distance from the center of the mass of the vehicle to the rear axles,
        self.dt = dt  # discretization time
        self.xbar = None  # linearization points
        self.ubar = None  # linearization points
        self.m = 2  # number of control inputs
        self.n = 4  # number of states
        self.length = lf + lr
        self.width = 0.5

    def set_lin_points(self, xbar, ubar):
        self.xbar = xbar
        self.ubar = ubar

    def _compute_beta_gamma(self):
        """β is the angle of the current velocity of the center of mass
             with respect to the longitudinal axis of the vehicle. """
        alpha = self.lr / (self.lf + self.lr)
        beta = np.arctan(alpha * np.tan(self.ubar[1]))
        gamma = alpha / ((np.cos(self.ubar[1]) ** 2) * (1 + (alpha ** 2) * (np.tan(self.ubar[1]) ** 2)))
        return beta, gamma

    def _compute_xbar_dot(self):
        beta, _ = self._compute_beta_gamma()
        xbar_dot = [[float(self.xbar[3] * np.cos(self.xbar[2] + beta))],
                    [float(self.xbar[3] * np.sin(self.xbar[2] + beta))],
                    [float(self.xbar[3] * np.sin(beta) / self.lr)],
                    [float(self.ubar[0])]]
        xbar_dot = np.array(xbar_dot, dtype=float)
        return xbar_dot

    def f_a(self):
        beta, _ = self._compute_beta_gamma()
        a_mat = [[0, 0, -self.xbar[3] * np.sin(self.xbar[2] + beta), np.cos(self.xbar[2] + beta)],
                 [0, 0, self.xbar[3] * np.cos(self.xbar[2] + beta), np.sin(self.xbar[2] + beta)],
                 [0, 0, 0, np.sin(beta) / self.lr],
                 [0, 0, 0, 0]]
        a_mat = np.array(a_mat, dtype=float)
        return a_mat

    def f_b(self):
        beta, gamma = self._compute_beta_gamma()
        b_mat = [[0, -gamma * self.xbar[3] * np.sin(self.xbar[2] + beta)],
                 [0, gamma * self.xbar[3] * np.cos(self.xbar[2] + beta)],
                 [0, (self.xbar[3] / self.lr) * gamma * np.cos(beta)],
                 [1, 0]]
        b_mat = np.array(b_mat, dtype=float)
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

    def linearize_at(self, xbar, ubar):
        self.model.set_lin_points(xbar, ubar)
        a_mat = self.model.f_a()
        b_mat = self.model.f_b()
        d_vec = self.model.f_d()
        return a_mat, b_mat, d_vec

    def update_states(self, x, u, xbar, ubar):
        a_mat, b_mat, d_vec = self.linearize_at(xbar, ubar)
        x = x + self.dt * (a_mat @ x + b_mat @ u + d_vec)
        x = np.array([float(item) for item in x]).reshape(-1, 1)
        return x


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
        """ this function is used in MPC loop to update the
         system states based on the optimization output """
        beta = self._compute_beta(u)
        f = [[float(x[3] * np.cos(x[2] + beta))],
             [float(x[3] * np.sin(x[2] + beta))],
             [float(x[3] * np.sin(beta) / self.lr)],
             [float(u[0])]]
        f = np.array(f)
        x = x + self.dt * f
        return x

    def _compute_beta_casadi(self, u):
        """ here, we used CasADi arctan for computing β instead of Numpy arctan"""
        alpha = self.lr / (self.lf + self.lr)
        beta = cs.arctan(alpha * cs.tan(u[1]))
        return beta

    def update_nls_states_casadi(self, x, u):
        """ nonlinear bicycle model,I wrote this function to check the nonlinear MPCC """
        beta = self._compute_beta_casadi(u)
        f = [x[3] * cs.cos(x[2] + beta),
             x[3] * cs.sin(x[2] + beta),
             x[3] * cs.sin(beta) / self.lr,
             u[0]]
        f = cs.vcat(f)
        return f