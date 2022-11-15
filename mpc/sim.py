import copy
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from data.routes import ThetaFinder, Trajectory, IntersectionLayout
import casadi as cs
import drawnow


class Polytope:
    def __init__(self, veh_len, veh_width):
        self.veh_len = veh_len
        self.veh_width = veh_width

    def get_polytope_A_b(self, x, y, orientation):
        row_1 = cs.hcat([cs.cos(orientation), -cs.sin(orientation)])
        row_2 = cs.hcat([cs.sin(orientation), cs.cos(orientation)])
        R = cs.vcat([row_1, row_2])
        R = cs.transpose(R)  # new

        A_poly = cs.vcat([R, -R])

        b_aux = np.array([self.veh_len / 2, self.veh_width / 2, self.veh_len / 2, self.veh_width / 2]).reshape(-1, 1)
        b_poly = b_aux + cs.mtimes(A_poly, cs.vcat([x, y]))
        return A_poly, b_poly


class SimParams:
    def __init__(self):
        self.N = 5  # prediction horizon
        self.tf = 20  # final time
        self.d_safe = 0.5  # safety distance
        self.qc = 1  # scalar
        self.ql = 1  # scalar
        self.q_theta = 1  # scalar
        self.Ru = 1  # matrix (m,1)
        self.Rv = 1  # scalar
        self.vx0 = 0.1  # initial vehicle speed in x-axis


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


class Optimization:
    """ class Optimization will be used in class Simulator. So, all traj, that is one of the inputs
     of class Optimization, comes from method set_vehicle_initial_conditions of class Simulator.
    """

    def __init__(self, params: SimParams, sys: LinearSystem, theta_finder: ThetaFinder):
        self.s_vars = None
        self.lambdas = None
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
        self.ca_vars = None
        self.objective = None
        self.polytope = Polytope(self.sys.model.length, self.sys.model.width)
        # Particle model, that is a linear model with 4 states and 2 inputs
        # self.A = np.array(
        #     [
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #         [0, 0, 0, 0],
        #         [0, 0, 0, 0]
        #     ]
        # )
        # self.B = np.array(
        #     [
        #         [0, 0],
        #         [0, 0],
        #         [1, 0],
        #         [0, 1]
        #     ]
        # )

    def set_all_traj(self, all_traj):
        self.all_traj = all_traj
        self.num_veh = len(all_traj)

    def set_vars(self):
        self.opti = cs.Opti()
        self.states = [self.opti.variable(self.sys.n, self.params.N + 1) for _ in range(self.num_veh)]
        self.theta = [self.opti.variable(1, self.params.N + 1) for _ in range(self.num_veh)]
        self.inputs = [self.opti.variable(self.sys.m, self.params.N) for _ in range(self.num_veh)]
        self.vir_inputs = [self.opti.variable(1, self.params.N) for _ in range(self.num_veh)]

        self.lambdas = [[[]] * self.num_veh for i in range(self.num_veh)]
        for i in range(self.num_veh):
            for j in range(self.num_veh):
                self.lambdas[i][j] = self.opti.variable(self.sys.n, self.params.N)

        self.s_vars = [[[]] * self.num_veh for i in range(self.num_veh)]
        for i in range(self.num_veh):
            for j in range(self.num_veh):
                self.s_vars[i][j] = self.opti.variable(self.sys.m, self.params.N)

    def _compute_phi_gamma(self, theta_bar, traj: Trajectory):
        phi = np.arctan2(traj.d_lut_y(theta_bar, 0), traj.d_lut_x(theta_bar, 0))
        gamma_1 = (traj.dd_lut_y(theta_bar, 0, 0) * traj.d_lut_x(theta_bar, 0)) - \
                  (traj.d_lut_y(theta_bar, 0) * traj.dd_lut_x(theta_bar, 0, 0))
        # aux = (traj.d_lut_y(theta_bar, 0) / (traj.d_lut_x(theta_bar, 0) + 1e-8)) + 1e-8
        gamma_2 = traj.d_lut_y(theta_bar, 0) ** 2 + traj.d_lut_x(theta_bar, 0) ** 2
        gamma = gamma_1.elements()[0] / (gamma_2.elements()[0] + 1e-8)

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
        """ Linearized obj """
        total_obj = 0
        for k in range(self.num_veh):
            obj = 0
            for i in range(1, self.params.N + 1):
                ec_bar, nabla_ec_bar, d_p_c, el_bar, nabla_el_bar, d_p_l = self._compute_contouring_lag_constants(
                    x_pred_all[k][:, i].reshape(-1, 1),
                    float(theta_pred_all[k][i]),
                    self.all_traj[k]
                )
                ec = ec_bar - cs.dot(nabla_ec_bar, x_pred_all[k][:, i].reshape(-1, 1)) + \
                     cs.dot(nabla_ec_bar, self.states[k][:, i]) \
                     + d_p_c * self.theta[k][i] - d_p_c * float(theta_pred_all[k][i])

                el = el_bar - cs.dot(nabla_el_bar, x_pred_all[k][:, i].reshape(-1, 1)) \
                     + cs.dot(nabla_el_bar, self.states[k][:, i]) \
                     + d_p_l * self.theta[k][i] - d_p_l * float(theta_pred_all[k][i])

                # """ Nonlinear Obj """
                # phi = cs.arctan2(self.all_traj[k].d_lut_y(self.theta[k][i], 0),
                #                  self.all_traj[k].d_lut_x(self.theta[k][i], 0))
                #
                # ec = cs.sin(phi) * (self.states[k][0, i] - self.all_traj[k].lut_x(self.theta[k][i])) - \
                #      cs.cos(phi) * (self.states[k][1, i] - self.all_traj[k].lut_y(self.theta[k][i]))
                #
                # el = - cs.cos(phi) * (self.states[k][0, i] - self.all_traj[k].lut_x(self.theta[k][i])) - \
                #      cs.sin(phi) * (self.states[k][1, i] - self.all_traj[k].lut_y(self.theta[k][i]))
                if i == 1:
                    delta_u = self.inputs[k][:, i - 1]
                    delta_vir_input = self.vir_inputs[k][i - 1]
                else:
                    delta_u = self.inputs[k][:, i - 1] - self.inputs[k][:, i - 2]
                    delta_vir_input = self.vir_inputs[k][i - 1] -  self.vir_inputs[k][i - 2]

                obj += self.params.qc * ec ** 2 + self.params.ql * el ** 2 - self.params.q_theta * self.theta[k][i] + \
                       cs.dot(delta_u, cs.mtimes(self.params.Ru, delta_u)) + \
                       self.params.Rv * delta_vir_input ** 2
            total_obj += obj
        self.objective = total_obj

    def set_constrs(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        # nl = NonlinearSystem(self.sys.dt, self.sys.model.lr, self.sys.model.lf)
        for k in range(self.num_veh):
            for i in range(1, self.params.N + 1):
                # Linearized system constraints
                a_mat, b_mat, d_vec = self.sys.linearize_at(
                    x_pred_all[k][:, i].reshape(-1, 1),
                    u_pred_all[k][:, i - 1].reshape(-1, 1)
                )
                self.opti.subject_to(
                    self.states[k][:, i] == self.states[k][:, i - 1] + self.sys.dt * (
                            cs.mtimes(a_mat, self.states[k][:, i - 1]) +
                            cs.mtimes(b_mat, self.inputs[k][:, i - 1]) +
                            d_vec)
                )


                # Progress constraints
                self.opti.subject_to(
                    self.theta[k][i] == self.theta[k][i - 1] + self.vir_inputs[k][i - 1]
                )

                # self.opti.subject_to(self.theta[k][i] >= 0)
                self.opti.subject_to(self.states[k][3, i] >= 0)  # minimum speed
                self.opti.subject_to(self.states[k][3, i] <= 10)  # maximum speed

            for _i in range(self.params.N):
                # self.opti.subject_to(self.vir_inputs[k][_i] >= 0)
                self.opti.subject_to(self.inputs[k][1, _i] >= -0.5)  # minimum steering angle
                self.opti.subject_to(self.inputs[k][1, _i] <= 0.5)  # maximum steering angle
                self.opti.subject_to(self.inputs[k][0, _i] >= -5)  # minimum acceleration
                self.opti.subject_to(self.inputs[k][0, _i] <= 5)  # maximum acceleration


            # initial condition constraints
            self.opti.subject_to(self.states[k][:, 0] == x_prev_all[k])
            self.opti.subject_to(self.theta[k][0] == theta_prev_all[k])


    def set_v2v_constrs(self):
        for i in range(self.num_veh):
            for j in range(self.num_veh):
                if j != i:
                    for k in range(self.params.N):

                        poly_a, poly_b = self.polytope.get_polytope_A_b(self.states[i][0, k], self.states[i][1, k],
                                                                    self.states[i][2, k])

                        poly_a_neighbour, poly_b_neighbour = self.polytope.get_polytope_A_b(self.states[j][0, k],
                                                                                            self.states[j][1, k],
                                                                                            self.states[j][2, k])
                        self.opti.subject_to(
                            (-cs.dot(poly_b, self.lambdas[i][j][:, k]) - cs.dot(poly_b_neighbour,
                                                                                self.lambdas[j][i][:,
                                                                                k])) >= self.params.d_safe
                        )

                        self.opti.subject_to(
                            cs.mtimes(cs.transpose(poly_a), self.lambdas[i][j][:, k]) + self.s_vars[i][j][:, k] == 0
                        )
                        self.opti.subject_to(
                            cs.mtimes(cs.transpose(poly_a_neighbour), self.lambdas[j][i][:, k]) - self.s_vars[i][j][:,
                                                                                                  k] == 0
                        )

                        self.opti.subject_to(self.lambdas[i][j][:, k] >= 0)
                        self.opti.subject_to(self.lambdas[j][i][:, k] >= 0)

                        expression = 0
                        for num_inputs in range(self.sys.m):
                            expression += cs.power(self.s_vars[i][j][num_inputs, :], 2)

                        self.opti.subject_to(
                            # cs.norm_2(self.s_vars[i][j][:, k]) <= 1
                            expression <= 1

                        )

                # for j in range(self.num_veh):
                #     if j != i:
                #         poly_a_neighbour, poly_b_neighbour = self.polytope.get_polytope_A_b(self.states[j][0, i],
                #                                                                             self.states[j][1, i],
                #                                                                             self.states[j][2, i])
                #         self.opti.subject_to(
                #             cs.dot(-poly_b, self.lambdas[i][l][:, i]) - cs.dot(poly_b_neighbour,
                #                                                                self.lambdas[j][i][:, i])
                #         )

    def solve(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        self.set_vars()
        self.set_obj(x_pred_all, theta_pred_all)
        self.opti.minimize(self.objective)
        self.set_constrs(x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all)
        # self.set_v2v_constrs()
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt'.lower(), opts)
        # self.opti.solver('qpOASES'.lower())
        solution = self.opti.solve()
        x_pred_all = [np.array(solution.value(self.states[i])).reshape(self.sys.n, self.params.N + 1) for i in
                      range(self.num_veh)]
        theta_pred_all = [np.array(solution.value(self.theta[i])).reshape(self.params.N + 1, 1) for i in
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
        self.opt.set_all_traj(self.all_traj)  # here, I assign value to all_traj that belongs to class Optimization.

    def optimize(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        return self.opt.solve(x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all)

    def update_vehicles_states(self, x_prev_list, u_opt_list, x_bar_list, u_bar_list):
        updated_x = []
        updated_theta = []
        sys_nl = NonlinearSystem(self.sys.dt, self.sys.model.lr, self.sys.model.lf)

        for i in range(self.num_veh):
            # x = self.sys.update_states(x_prev_list[i], u_opt_list[i], x_bar_list[i], u_bar_list[i])  # based on linearized model
            x = sys_nl.update_nls_states(x_prev_list[i], u_opt_list[i])  # based on nonlinear model
            theta = self.theta_finder.find_theta(x[0], x[1])
            updated_x.append(x)
            updated_theta.append(theta)
        return updated_x, updated_theta

    def _get_prediction(self, x0, theta0, traj):
        x_pred = np.tile(x0, (1, self.params.N + 1))
        theta_pred = np.tile(theta0, (self.params.N + 1, 1))
        for i in range(1, self.params.N + 1):
            theta_next = theta_pred[i - 1] + self.params.vx0

            phi_next = np.arctan2(traj.d_lut_y(theta_next, 0), traj.d_lut_x(theta_next, 0))

            if (x_pred[2, i - 1] - phi_next) < - np.pi:
                phi_next = phi_next - 2 * np.pi

            if (x_pred[2, i - 1] - phi_next) > np.pi:
                phi_next = phi_next + 2 * np.pi

            x_pred[:, i] = np.array([[traj.lut_x(theta_next)],
                                     [traj.lut_y(theta_next)],
                                     [phi_next],
                                     [self.params.vx0]], dtype=float).reshape(-1, )

            theta_pred[i] = theta_next

        return x_pred, theta_pred

    def get_prediction_all_vehicles(self):
        x_pred_all = []
        theta_pred_all = []
        for i in range(self.num_veh):
            x_pred, theta_pred = self._get_prediction(self.x_init_list[i], self.theta_init_list[i], self.all_traj[i])
            x_pred_all.append(x_pred)
            theta_pred_all.append(theta_pred)

        u_pred_all = [np.zeros((self.sys.m, self.params.N)) for _ in range(self.num_veh)]
        u_vir_pred_all = [np.zeros((self.params.N, 1)) for _ in range(self.num_veh)]
        return x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all

    # def _get_shift_prediction(self, x_pred, theta_pred, u_pred, u_vir_pred):
    #     # todo: replace last zero by simulating the nonlinear system based on inputs and states from step N-1
    #     x_pred_shifted = np.concatenate((x_pred[:, 1:], np.zeros((self.sys.n, 1))), axis = 1)
    #     theta_pred_shifted = np.concatenate((theta_pred[1:], np.zeros((1, 1))), axis = 0)
    #     u_pred_shifted = np.concatenate((u_pred[:, 1:], np.zeros((self.sys.m, 1))), axis = 1)
    #     u_vir_pred_shifted = np.concatenate((u_vir_pred[1:], np.zeros((1, 1))), axis = 0)
    #     return x_pred_shifted, theta_pred_shifted, u_pred_shifted, u_vir_pred_shifted
    def _get_shift_prediction(self, x_pred, theta_pred, u_pred, u_vir_pred, current_x, current_theta, traj):
        n = self.sys.n
        m = self.sys.m
        N = self.params.N

        x_temp = np.zeros((n, N + 1))
        theta_temp = np.zeros((N + 1, 1))
        u_temp = np.zeros((m, N))
        u_vir_temp = np.zeros((N, 1))

        x_temp[:, 0] = current_x.T
        theta_temp[0] = current_theta
        u_temp[:, 0] = u_pred[:, 1]
        u_vir_temp[0] = u_vir_pred[1]

        for i in range(1, N - 1):
            x_temp[:, i] = x_pred[:, i + 1]
            theta_temp[i] = theta_pred[i + 1]
            u_temp[:, i] = u_pred[:, i + 1]
            u_vir_temp[i] = u_vir_pred[i + 1]

        i = N - 1
        x_temp[:, i] = x_pred[:, i + 1]
        theta_temp[i] = theta_pred[i + 1]
        u_temp[:, i] = u_pred[:, i]
        u_vir_temp[i] = u_vir_pred[i]

        nl = NonlinearSystem(self.sys.dt, self.sys.model.lr, self.sys.model.lf)
        i = N
        x_temp[:, i] = nl.update_nls_states(x_pred[:, N].reshape(-1, 1), u_pred[:, N - 1].reshape(-1, 1)).T
        theta_temp[i] = theta_pred[N] + u_vir_pred[N - 1]

        # --- New: begin
        # cl = traj.cl[-1]
        if (x_temp[2, 0] - x_temp[2, 1]) > np.pi:
            x_temp[2, 1:] = x_temp[2, 1:] + 2 * np.pi
        if (x_temp[2, 0] - x_temp[2, 1]) < -np.pi:
            x_temp[2, 1:] = x_temp[2, 1:] - 2 * np.pi
        # if (x_temp[2, 0] - x_temp[2, 1]) < - 0.75 * cl:
        #     x_temp[2, 1:] = x_temp[2, 1:] - cl
        # --- New: end

        return x_temp, theta_temp, u_temp, u_vir_temp

    def get_shift_prediction_all_vehicles(self, x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all, current_x,
                                          current_theta):
        x_pred_shifted_all = []
        theta_pred_shifted_all = []
        u_pred_shifted_all = []
        u_vir_pred_shifted_all = []
        for i in range(self.num_veh):
            x_pred_shifted, theta_pred_shifted, u_pred_shifted, u_vir_pred_shifted = \
                self._get_shift_prediction(x_pred_all[i], theta_pred_all[i], u_pred_all[i], u_vir_pred_all[i],
                                           current_x[i], current_theta[i], self.all_traj[i])
            x_pred_shifted_all.append(x_pred_shifted)
            theta_pred_shifted_all.append(theta_pred_shifted)
            u_pred_shifted_all.append(u_pred_shifted)
            u_vir_pred_shifted_all.append(u_vir_pred_shifted)

        return x_pred_shifted_all, theta_pred_shifted_all, u_pred_shifted_all, u_vir_pred_shifted_all

    def _unwrap_x0(self, x0):
        if x0[2] > np.pi:
            x0[2] = x0[2] - 2 * np.pi
        if x0[2] <= - np.pi:
            x0[2] = x0[2] + 2 * np.pi
        return x0

    def get_unwrap_all_vehicles(self, x):
        x_all_unwrap = []
        for i in range(self.num_veh):
            x0 = self._unwrap_x0(x[i])
            x_all_unwrap.append(x0)

        return x_all_unwrap

    def run(self):
        XX = [[] for i in range(self.num_veh)]  # defined for saving x[0]
        YY = [[] for i in range(self.num_veh)]  # defined for saving x[1]
        speed = [[] for i in range(self.num_veh)]
        XX_pred = [[] for i in range(self.num_veh)]
        YY_pred = [[] for i in range(self.num_veh)]
        time = np.arange(0, self.params.tf, self.sys.dt)
        x = self.x_init_list
        theta = self.theta_init_list
        u = self.u_init_list
        # write predictions for x, theta, and u
        x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all = self.get_prediction_all_vehicles()

        intersection = IntersectionLayout(self.theta_finder.track, self.theta_finder.track.lane_width, 150)

        ## drawnow code
        def draw_fig():
            # plt.show()
            plt.subplot(211)
            intersection.plot_intersection()
            for i in range(self.num_veh):
                plt.subplot(211)
                plt.plot(XX[i], YY[i], label=f"veh_{i}")
                plt.plot(XX_pred[i], YY_pred[i], '--')
                plt.legend()
                plt.subplot(212)
                plt.plot(speed[i], label=f"veh_{i}")
                plt.legend()

        fig, ax = plt.subplots()
        # MPC loop
        for t_ind, t in enumerate(time):
            xbar = copy.deepcopy(x)
            ubar = copy.deepcopy(u)

            # obtaining predictions of states and inputs for linearizing model and objective function
            x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all = \
                self.get_shift_prediction_all_vehicles(x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all, x, theta)

            x_pred_all, theta_pred_all, u_pred_all, u_vir_pred_all = self.optimize(x,
                                                                                   theta,
                                                                                   x_pred_all,
                                                                                   theta_pred_all,
                                                                                   u_pred_all
                                                                                   )
            # for i in range(self.num_veh):
            #     print(f"veh_{i}: {u_vir_pred_all[i][0]}")

            for i in range(self.num_veh):
                print(f"veh_{i}: {theta_pred_all[i][0]}")

            u = [upred[:, 0].reshape(-1, 1) for upred in u_pred_all]

            x, theta = self.update_vehicles_states(x, u, xbar, ubar)
            x = self.get_unwrap_all_vehicles(x)  # I wrote function "unwrap_x0(x)", based on Liniger's code

            for i in range(self.num_veh):
                XX[i].append(x[i][0])
                YY[i].append(x[i][1])
                speed[i].append(x[i][3])
                drawnow.drawnow(draw_fig, stop_on_close=True)
                XX_pred[i] = x_pred_all[i][0, :]
                YY_pred[i] = x_pred_all[i][1, :]
        return XX, YY, self.all_traj, XX_pred, YY_pred

    def get_results(self):

        pass
