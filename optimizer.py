from typing import Union, List

import casadi as cs
import numpy as np

from routes import ThetaFinder, Trajectory
from system import LinearSystem, NonlinearSystem


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


class Optimization:
    """ class Optimization will be used in class Simulator. So, all traj, that is one of the inputs
     of class Optimization, comes from method set_vehicle_initial_conditions of class Simulator.
    """

    def __init__(self, params, sys: LinearSystem, theta_finder: ThetaFinder):
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
                    delta_vir_input = self.vir_inputs[k][i - 1] - self.vir_inputs[k][i - 2]

                obj += self.params.qc * ec ** 2 + self.params.ql * el ** 2 - self.params.q_theta * self.theta[k][i] + \
                       cs.dot(delta_u, cs.mtimes(self.params.Ru, delta_u)) + \
                       self.params.Rv * delta_vir_input ** 2
            total_obj += obj
        self.objective = total_obj

    def set_constrs(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        nl = NonlinearSystem(self.sys.dt, self.sys.model.lr, self.sys.model.lf)
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

                # Nonlinear system constraints
                # self.opti.subject_to(
                #     self.states[k][:, i] == self.states[k][:, i - 1] + self.sys.dt * nl.update_nls_states_casadi(
                #         self.states[k][:, i - 1], self.inputs[k][:, i - 1]
                #     )
                # )

                # Particle system constraints
                # self.opti.subject_to(
                #     self.states[k][:, i] == self.states[k][:, i - 1] + self.sys.dt * (
                #             cs.mtimes(self.A, self.states[k][:, i - 1]) +
                #             cs.mtimes(self.B, self.inputs[k][:, i - 1])
                #     )
                # )

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

    def set_v2v_constrs_ball(self):
        for i in range(self.num_veh):
            for j in range(i + 1, self.num_veh):
                for k in range(self.params.N):
                    self.opti.subject_to(
                        (self.states[i][0, k] - self.states[j][0, k]) ** 2 +
                        (self.states[i][1, k] - self.states[j][1, k]) ** 2
                        >= (self.params.d_safe) ** 2
                    )

    def set_boundaries_constr_ball(self):
        for i in range(self.num_veh):
            for k in range(1, self.params.N + 1):
                self.opti.subject_to(
                    cs.sqrt((self.all_traj[i].lut_x(self.theta[i][k]) - self.states[i][0, k]) ** 2 +
                            (self.all_traj[i].lut_y(self.theta[i][k]) - self.states[i][1, k]) ** 2)
                    <= ((self.theta_finder.track.track_width - self.sys.model.width) / 2)
                )

    def solve(self, x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all):
        self.set_vars()
        self.set_obj(x_pred_all, theta_pred_all)
        self.opti.minimize(self.objective)
        self.set_constrs(x_prev_all, theta_prev_all, x_pred_all, theta_pred_all, u_pred_all)
        self.set_boundaries_constr_ball()
        # self.set_v2v_constrs()
        # self.set_v2v_constrs_ball()
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
