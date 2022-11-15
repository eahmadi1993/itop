import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from optimizer import Optimization
from routes import ThetaFinder, IntersectionLayout
import drawnow

from system import LinearSystem, NonlinearSystem


class MPCCParams:
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


class MPCC:
    def __init__(self, params: MPCCParams, sys: LinearSystem, theta_finder: ThetaFinder):
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
            self.theta_finder.set_initial_conditions(self.x_init_list[i][0], self.x_init_list[i][1])
            theta = self.theta_finder.find_theta(x[0], x[1])
            updated_x.append(x)
            updated_theta.append(theta)
        return updated_x, updated_theta

    # def _get_prediction(self, x0, theta0, traj):
    #     x_pred = np.tile(x0, (1, self.params.N + 1))
    #     theta_pred = np.tile(theta0, (self.params.N + 1, 1))
    #     for i in range(1, self.params.N + 1):
    #         theta_next = theta_pred[i - 1] + self.params.vx0
    #
    #         phi_next = np.arctan2(traj.d_lut_y(theta_next, 0), traj.d_lut_x(theta_next, 0))
    #
    #         if (x_pred[2, i - 1] - phi_next) < - np.pi:
    #             phi_next = phi_next - 2 * np.pi
    #
    #         if (x_pred[2, i - 1] - phi_next) > np.pi:
    #             phi_next = phi_next + 2 * np.pi
    #
    #         x_pred[:, i] = np.array([[traj.lut_x(theta_next)],
    #                                  [traj.lut_y(theta_next)],
    #                                  [phi_next],
    #                                  [self.params.vx0]], dtype = float).reshape(-1, )
    #
    #         theta_pred[i] = theta_next
    #
    #     return x_pred, theta_pred
    def _get_prediction(self, x0, theta0):
        nl = NonlinearSystem(self.sys.dt, self.sys.model.lr, self.sys.model.lf)
        x_pred = np.zeros((self.sys.n, self.params.N + 1))
        theta_pred = np.zeros((self.params.N + 1, 1))
        u = np.zeros((self.sys.m, self.params.N))
        for i in range(self.params.N + 1):

            if i == 0:
                x_pred[:, i] = x0.T
                theta_pred[i] = theta0
            else:
                x_pred[:, i] = nl.update_nls_states(x_pred[:, i - 1].reshape(-1, 1), u[:, i - 1].reshape(-1, 1)).T
                self.theta_finder.set_initial_conditions(x0[0], x0[1])
                theta_pred[i] = self.theta_finder.find_theta(x_pred[0, i], x_pred[1, i])
        return x_pred, theta_pred

    def get_prediction_all_vehicles(self):
        x_pred_all = []
        theta_pred_all = []
        for i in range(self.num_veh):
            # x_pred, theta_pred = self._get_prediction(self.x_init_list[i], self.theta_init_list[i], self.all_traj[i])
            x_pred, theta_pred = self._get_prediction(self.x_init_list[i], self.theta_init_list[i])
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

        # drawnow code
        def draw_fig():
            # plt.show()
            plt.subplot(211)
            intersection.plot_intersection()
            for i in range(self.num_veh):
                plt.subplot(211)
                plt.plot(XX[i], YY[i], label = f"veh_{i}")
                plt.plot(XX_pred[i], YY_pred[i], '--')
                plt.legend()
                plt.subplot(212)
                plt.plot(speed[i], label = f"veh_{i}")
                plt.legend()

        # fig, ax = plt.subplots()
        # MPC loop
        for t_ind, t in enumerate(time):

            # print(f"simulation progress: {t/time[-1] * 100:6.1f}%")
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

            # for i in range(self.num_veh):
            #     print(f"veh_{i}: {theta_pred_all[i][0]}")

            u = [upred[:, 0].reshape(-1, 1) for upred in u_pred_all]

            x, theta = self.update_vehicles_states(x, u, xbar, ubar)
            x = self.get_unwrap_all_vehicles(x)  # I wrote function "unwrap_x0(x)", based on Liniger's code
            print(x_pred_all[0][:, -1].T)
            for i in range(self.num_veh):
                XX[i].append(x[i][0])
                YY[i].append(x[i][1])
                speed[i].append(x[i][3])
                drawnow.drawnow(draw_fig, stop_on_close = True)
                XX_pred[i] = x_pred_all[i][0, :]
                YY_pred[i] = x_pred_all[i][1, :]
        return XX, YY, self.all_traj, XX_pred, YY_pred, speed

    def get_results(self):

        pass
