import copy
from typing import List, Union

import numpy as np
from data.routes import ThetaFinder, Trajectory
import casadi as cs


class SimParams:
    def __init__(self):
        self.N = 5  # prediction horizon
        self.tf = 20  # final time
        self.d_safe = 0.1  # safety distance
        self.qc = 1  # scalar
        self.ql = 1  # scalar
        self.q_theta = 1  # scalar
        self.Ru = 1  # (m,1)
        self.Rv = 1  # scalar
        self.vx0 = 0.5  #initial veicle speed in x-axis


class BicycleModel:
    def __init__(self, dt, lf, lr):
        self.lf = lf  # distance from the center of the mass of the vehicle to the front axles
        self.lr = lr  # distance from the center of the mass of the vehicle to the rear axles,
        self.dt = dt  # discretization time
        self.xbar = None  # linearization points
        self.ubar = None  # linearization points
        self.m = 2  # number of control inputs
        self.n = 4  # number of states

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
        xbar_dot = np.array(xbar_dot, dtype = float)
        return xbar_dot

    def f_a(self):
        beta, _ = self._compute_beta_gamma()
        a_mat = [[0, 0, -self.xbar[3] * np.sin(self.xbar[2] + beta), np.cos(self.xbar[2] + beta)],
                 [0, 0, self.xbar[3] * np.cos(self.xbar[2] + beta), np.sin(self.xbar[2] + beta)],
                 [0, 0, 0, np.sin(beta) / self.lr],
                 [0, 0, 0, 0]]
        a_mat = np.array(a_mat, dtype = float)
        return a_mat

    def f_b(self):
        beta, gamma = self._compute_beta_gamma()
        b_mat = [[0, -gamma * self.xbar[3] * np.sin(self.xbar[2] + beta)],
                 [0, gamma * self.xbar[3] * np.cos(self.xbar[2] + beta)],
                 [0, gamma * np.cos(beta)],
                 [1, 0]]
        b_mat = np.array(b_mat, dtype = float)
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


class Optimization:
    """ class Optimization will be used in class Simulator. So, all traj, that is one of the inputs
     of class Optimization, comes from method set_vehicle_initial_conditions of class Simulator.
    """

    def __init__(self, params: SimParams, sys: LinearSystem, theta_finder: ThetaFinder):
        self.params = params
        self.sys = sys
        self.theta_finder = theta_finder
        self.all_traj: Union[List[Trajectory], None] = None
        self.num_veh = 0

        self.opti = None
        self.states = None
        self.inputs = None
        self.vir_inputs = None
        self.theta = None

        self.objective = None

    def set_all_traj(self, all_traj):
        self.all_traj = all_traj
        self.num_veh = len(all_traj)

    def set_vars(self):
        self.opti = cs.Opti()
        self.states = [self.opti.variable(self.sys.n, self.params.N) for _ in range(self.num_veh)]
        self.theta = [self.opti.variable(1, self.params.N) for _ in range(self.num_veh)]
        self.inputs = [self.opti.variable(self.sys.m, self.params.N) for _ in range(self.num_veh)]
        self.vir_inputs = [self.opti.variable(1, self.params.N) for _ in range(self.num_veh)]

    def _compute_phi_gamma(self, theta_bar, traj: Trajectory):
        phi = np.arctan2(traj.d_lut_y(theta_bar, 0), traj.d_lut_x(theta_bar, 0))
        gamma_1 = (traj.dd_lut_y(theta_bar, 0, 0) * traj.d_lut_x(theta_bar, 0)) - \
                  (traj.d_lut_y(theta_bar, 0) * traj.dd_lut_x(theta_bar, 0, 0))
        gamma_2 = (1 + phi ** 2) * traj.dd_lut_x(theta_bar, 0, 0)
        gamma = gamma_1.elements()[0] / gamma_2.elements()[0]
        return phi, gamma

    def _compute_contouring_lag_constants(self, xbar, theta_bar, traj: Trajectory):
        phi_bar, gamma = self._compute_phi_gamma(theta_bar, traj)
        ec_bar = np.sin(phi_bar) * (xbar[0] - traj.lut_x(theta_bar)) - \
                 np.cos(phi_bar) * (xbar[1] - traj.lut_y(theta_bar))
        el_bar = -np.cos(phi_bar) * (xbar[0] - traj.lut_x(theta_bar)) - \
                 np.sin(phi_bar) * (xbar[1] - traj.lut_y(theta_bar))

        nabla_ec_bar = np.zeros((self.sys.n, 1))
        nabla_ec_bar[0] = np.sin(phi_bar)
        nabla_ec_bar[1] = -np.cos(phi_bar)

        nabla_el_bar = np.zeros((self.sys.n, 1))
        nabla_el_bar[0] = -np.cos(phi_bar)
        nabla_el_bar[1] = -np.sin(phi_bar)

        d_p_c = -gamma * el_bar - np.sin(phi_bar) * traj.d_lut_x(theta_bar, 0) + \
                np.cos(phi_bar) * traj.d_lut_y(theta_bar, 0)

        d_p_l = gamma * ec_bar + np.cos(phi_bar) * traj.d_lut_x(theta_bar, 0) + \
                np.sin(phi_bar) * traj.d_lut_y(theta_bar, 0)

        return ec_bar, nabla_ec_bar, d_p_c, el_bar, nabla_el_bar, d_p_l

    def set_obj(self, x_pred_all, theta_pred_all):

        total_obj = 0
        for k in range(self.num_veh):
            obj = 0
            for i in range(self.params.N):
                ec_bar, nabla_ec_bar, d_p_c, el_bar, nabla_el_bar, d_p_l = self._compute_contouring_lag_constants(
                    x_pred_all[k][:, i].reshape(-1, 1),
                    float(theta_pred_all[k][i]),
                    self.all_traj[k]
                )
                ec = ec_bar + cs.dot(nabla_ec_bar, self.states[k][:, i]) + d_p_c * self.theta[k][i]
                el = el_bar + cs.dot(nabla_el_bar, self.states[k][:, i]) + d_p_l * self.theta[k][i]
                obj += self.params.qc * ec ** 2 + \
                       self.params.ql * el ** 2 - \
                       self.params.q_theta * self.theta[k][i] + \
                       cs.dot(self.inputs[k][:, i], cs.mtimes(self.params.Ru, self.inputs[k][:, i])) + \
                       self.params.Rv * self.vir_inputs[k][i] ** 2

            total_obj += obj
        self.objective = total_obj

    def set_constrs(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        for k in range(self.num_veh):
            # system constraints
            for i in range(1, self.params.N):
                a_mat, b_mat, d_vec = self.sys.linearize_at(
                    x_pred_all[k][:, i].reshape(-1, 1),
                    u_pred_all[k][:, i].reshape(-1, 1)
                )
                self.opti.subject_to(
                    self.states[k][:, i] == self.states[k][:, i - 1] + self.sys.dt * (
                            cs.mtimes(a_mat, self.states[k][:, i - 1]) +
                            cs.mtimes(b_mat, self.inputs[k][:, i - 1]) +
                            d_vec)
                )
                self.opti.subject_to(
                    self.theta[k][i] == self.theta[k][i - 1] + self.vir_inputs[k][i]
                )
            # initial condition constraints
            self.opti.subject_to(self.states[k][:, 0] == x_prev_all[k])
            self.opti.subject_to(self.theta[k][0] == theta_prev_all[k])

    def solve(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        self.set_vars()
        self.set_obj(x_pred_all, theta_pred_all)
        self.opti.minimize(self.objective)
        self.set_constrs(x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all)

        self.opti.solver("ipopt")
        solution = self.opti.solve()
        x_pred_all = [np.array(solution.value(self.states[i])).reshape(self.sys.n, self.params.N) for i in
                      range(self.num_veh)]
        theta_pred_all = [np.array(solution.value(self.theta[i])).reshape(self.params.N, 1) for i in
                          range(self.num_veh)]
        u_pred_all = [np.array(solution.value(self.inputs[i])).reshape(self.sys.m, self.params.N) for i in
                      range(self.num_veh)]
        u_vir_pred_all = [np.array(solution.value(self.vir_inputs[i])).reshape(self.params.N, 1) for i in
                          range(self.num_veh)]
        return x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all


class Simulator:
    def __init__(self, params: SimParams, sys: LinearSystem, theta_finder: ThetaFinder):
        self.params = params
        self.sys = sys
        self.theta_finder = theta_finder
        self.opt = Optimization(self.params, self.sys, self.theta_finder)
        self.num_veh = None
        self.x_init_list = None
        self.theta_init_list = []
        self.u_init_list = []
        self.all_traj = []  # here, all_traj belongs to class Simulator.

    def set_vehicle_initial_conditions(self, x_init_list):
        self.num_veh = len(x_init_list)
        self.x_init_list = x_init_list
        m = self.sys.m
        for i in range(self.num_veh):
            self.theta_finder.set_initial_conditions(x_init_list[i][0], x_init_list[i][1])
            self.all_traj.append(self.theta_finder.mytraj)
            self.u_init_list.append(np.zeros((m, 1)))
            self.theta_init_list.append(
                self.theta_finder.find_theta(self.x_init_list[i][0], self.x_init_list[i][1])
            )
        self.opt.set_all_traj(self.all_traj) # here, I assign value to all_traj that belongs to class Optimization.

    def optimize(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        return self.opt.solve(x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all)

    def update_vehicles_states(self, x_prev_list, u_opt_list, x_bar_list, u_bar_list):
        updated_x = []
        updated_theta = []
        for i in range(self.num_veh):
            x = self.sys.update_states(x_prev_list[i], u_opt_list[i], x_bar_list[i], u_bar_list[i])
            theta = self.theta_finder.find_theta(x[0], x[1])
            updated_x.append(x)
            updated_theta.append(theta)
        return updated_x, updated_theta

    def get_prediction(self):
        x0 = 1
        theta0 = self.theta_init_list
        x_pred = np.tile(x0, 1, self.params.N+1)
        theta_pred = np.tile(theta0, self.params.N+1, 1)

        for i in range(2, self.params.N+1):
            theta_pred[i] = theta0[i-1] + self.sys.dt * self.params.vx0

        u_pred = np.zeros((self.sys.m, self.params.N))
        u_vir_pred = np.zeros((self.params.N, 1))
        return u_pred, u_vir_pred, theta_pred


    def run(self):
        time = np.arange(0, self.params.tf, self.sys.dt)
        x = self.x_init_list
        theta = self.theta_init_list
        u = self.u_init_list
        # TODO: write predictions for x, theta, and u
        x_pred_all = [np.zeros((self.sys.n, self.params.N))]
        theta_pred_all = [np.zeros((self.params.N, 1))]
        u_pred_all = [np.zeros((self.sys.m, self.params.N))]

        for t_ind, t in enumerate(time):  # MPC loop
            xbar = copy.deepcopy(x)
            ubar = copy.deepcopy(u)

            x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all = self.optimize(x, theta, x_pred_all, theta_pred_all,
                                                                                   u_pred_all)
            u = [upred[:, 0].reshape(-1,1) for upred in u_pred_all]

            x, theta = self.update_vehicles_states(x, u, xbar, ubar)
            # TODO: I should write function "x{k} = unWrapX0(x{k})", based on Liniger's code

    def get_results(self):
        pass
