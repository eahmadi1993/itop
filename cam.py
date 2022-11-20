from itertools import accumulate
from typing import List
import uuid
import numpy as np
from routes import ThetaFinder, IntersectionLayout
from sim_mpc import MPCC, MPCCParams
from system import LinearSystem
import drawnow
import matplotlib.pyplot as plt


class Vehicle:
    def __init__(self, initial_condition):
        self.id = str(uuid.uuid4())
        self.current_states = None
        self.initial_condition = initial_condition
        if initial_condition[2] == np.pi:
            self.orientation = "east"
        elif initial_condition[2] == 0:
            self.orientation = "west"
        elif initial_condition[2] == np.pi / 2:
            self.orientation = "south"
        else:  # -np.pi/2
            self.orientation = "north"
        print(f"vehicle approaching from {self.orientation} with {initial_condition.T}")


class VehicleManager:

    def __init__(self):
        self.west_vehicles_list: List[Vehicle] = []
        self.east_vehicles_list: List[Vehicle] = []
        self.north_vehicles_list: List[Vehicle] = []
        self.south_vehicles_list: List[Vehicle] = []

    def _remove_vehicle_from_list(self, veh_list: List[Vehicle], vehicle: Vehicle):
        for ind, veh in enumerate(veh_list):
            if veh.id == vehicle.id:
                veh_list.pop(ind)
            return veh_list

    def add_veh_west(self, veh: Vehicle):
        self.west_vehicles_list.append(veh)

    def add_veh_east(self, veh: Vehicle):
        self.east_vehicles_list.append(veh)

    def add_veh_north(self, veh: Vehicle):
        self.north_vehicles_list.append(veh)

    def add_veh_south(self, veh: Vehicle):
        self.south_vehicles_list.append(veh)

    def remove_veh_west(self, vehicle: Vehicle):
        self.west_vehicles_list = self._remove_vehicle_from_list(self.west_vehicles_list, vehicle)

    def remove_veh_east(self, vehicle: Vehicle):
        self.east_vehicles_list = self._remove_vehicle_from_list(self.east_vehicles_list, vehicle)

    def remove_veh_north(self, vehicle: Vehicle):
        self.north_vehicles_list = self._remove_vehicle_from_list(self.north_vehicles_list, vehicle)

    def remove_veh_south(self, vehicle: Vehicle):
        self.south_vehicles_list = self._remove_vehicle_from_list(self.south_vehicles_list, vehicle)

    def intersection_is_empty(self):
        if len(self.west_vehicles_list) == 0 and len(self.east_vehicles_list) == 0 and len(
                self.north_vehicles_list) == 0 and len(self.south_vehicles_list) == 0:
            return True

        return False


class Simulator:
    def __init__(self, params: MPCCParams, system: LinearSystem, theta_finder: ThetaFinder):
        # todo: manage the parameters and the constructor

        self.mpcc = MPCC(params, system, theta_finder)
        self.sim_steps = None
        self.arrivals = 3  # subject to change
        self.flow = 600  # subject to change
        self.rng = np.random.default_rng()
        self.west_arrival = None
        self.south_arrival = None
        self.east_arrival = None
        self.north_arrival = None
        self.additional_steps = 500
        self.set_arrivals()

        self.vehicle_manager = VehicleManager()
        self.x_init = []
        self.x_pred_all = []
        self.theta_pred_all = []
        self.u_pred_all = []
        self.vir_pred_all = []

    def set_arrivals(self):
        headway = 3600 / self.flow

        self.west_arrival = np.bincount(list(accumulate(self.rng.poisson(headway, self.arrivals) / self.mpcc.sys.dt)))
        self.south_arrival = np.bincount(list(accumulate(self.rng.poisson(headway, self.arrivals) / self.mpcc.sys.dt)))
        self.east_arrival = np.bincount(list(accumulate(self.rng.poisson(headway, self.arrivals) / self.mpcc.sys.dt)))
        self.north_arrival = np.bincount(list(accumulate(self.rng.poisson(headway, self.arrivals) / self.mpcc.sys.dt)))

        self.sim_steps = max(
            len(self.west_arrival),
            len(self.south_arrival),
            len(self.east_arrival),
            len(self.north_arrival),
        ) + self.additional_steps

    def run_simulation(self):
        x = []
        theta = []

        def handle_vehicle_entrance(x, theta):

            self.x_init.append(x0)
            self.mpcc.set_vehicle_initial_conditions(self.x_init)

            x_pred, theta_pred = self.mpcc.get_prediction(x0, self.mpcc.theta_init_list[-1], self.mpcc.all_traj[-1])
            u_pred = np.zeros((self.mpcc.sys.m, self.mpcc.params.N))

            self.x_pred_all.append(x_pred)
            self.u_pred_all.append(u_pred)
            self.theta_pred_all.append(theta_pred)
            vir_pred = np.zeros((self.mpcc.params.N, 1))
            self.vir_pred_all.append(vir_pred)
            if self.vehicle_manager.intersection_is_empty():
                x = self.mpcc.x_init_list
                theta = self.mpcc.theta_init_list
            else:
                x.append(x0)
                theta.append(self.mpcc.theta_init_list[-1])
            return x, theta

        for i in range(self.sim_steps):
            try:
                if self.north_arrival[i] == 1:
                    x0 = np.array([self.rng.uniform(23, 32.1), 60, -np.pi / 2, self.rng.normal(2)],
                                  dtype = float).reshape(
                        -1,
                        1)
                    x, theta = handle_vehicle_entrance(x, theta)
                    XX = [[] for i in range(self.mpcc.num_veh)]
                    YY = [[] for i in range(self.mpcc.num_veh)]
                    speed = [[] for i in range(self.mpcc.num_veh)]
                    self.vehicle_manager.add_veh_north(Vehicle(
                        x0
                    ))
                if self.south_arrival[i] == 1:
                    x0 = np.array([self.rng.uniform(32.1, 41), 0, np.pi / 2, self.rng.normal(2)],
                                  dtype = float).reshape(-1,
                                                         1)
                    x, theta = handle_vehicle_entrance(x, theta)
                    XX = [[] for i in range(self.mpcc.num_veh)]
                    YY = [[] for i in range(self.mpcc.num_veh)]
                    speed = [[] for i in range(self.mpcc.num_veh)]
                    self.vehicle_manager.add_veh_south(Vehicle(
                        x0
                    ))
                if self.east_arrival[i] == 1:
                    x0 = np.array([60, self.rng.uniform(29.2, 41), np.pi, self.rng.normal(2)], dtype = float).reshape(
                        -1,
                        1)
                    x, theta = handle_vehicle_entrance(x, theta)
                    XX = [[] for i in range(self.mpcc.num_veh)]
                    YY = [[] for i in range(self.mpcc.num_veh)]
                    speed = [[] for i in range(self.mpcc.num_veh)]
                    self.vehicle_manager.add_veh_east(Vehicle(
                        x0
                    ))
                if self.west_arrival[i] == 1:
                    x0 = np.array([0, self.rng.uniform(20, 29), 0, self.rng.normal(2)], dtype = float).reshape(-1, 1)

                    x, theta = handle_vehicle_entrance(x, theta)
                    XX = [[] for i in range(self.mpcc.num_veh)]
                    YY = [[] for i in range(self.mpcc.num_veh)]
                    speed = [[] for i in range(self.mpcc.num_veh)]
                    self.vehicle_manager.add_veh_west(Vehicle(
                        x0
                    ))
            except Exception as exp:
                print(exp)
                pass

            if self.vehicle_manager.intersection_is_empty():
                continue

            self.x_pred_all, self.theta_pred_all, self.u_pred_all, self.vir_pred_all = self.mpcc.optimize(x,
                                                                                                          theta,
                                                                                                          self.x_pred_all,
                                                                                                          self.theta_pred_all,
                                                                                                          self.u_pred_all
                                                                                                          )
            u = [upred[:, 0].reshape(-1, 1) for upred in self.u_pred_all]
            x, theta = self.mpcc.update_vehicles_states(x, u)
            x = self.mpcc.get_unwrap_all_vehicles(x)

            self.x_pred_all, self.theta_pred_all, self.u_pred_all, self.vir_pred_all = \
                self.mpcc.get_shift_prediction_all_vehicles(self.x_pred_all,
                                                            self.theta_pred_all,
                                                            self.u_pred_all,
                                                            self.vir_pred_all,
                                                            x,
                                                            theta)
            intersection = IntersectionLayout(self.mpcc.theta_finder.track, self.mpcc.theta_finder.track.lane_width,
                                              150)

            def draw_fig():
                # plt.show()
                plt.subplot(211)
                intersection.plot_intersection()
                for ii in range(self.mpcc.num_veh):
                    plt.subplot(211)
                    plt.plot(XX[ii], YY[ii])
                    plt.subplot(212)
                    plt.plot(speed[ii])


            for ii in range(self.mpcc.num_veh):
                XX[ii].append(x[ii][0])
                YY[ii].append(x[ii][1])
                speed[ii].append(x[ii][3])
                drawnow.drawnow(draw_fig, stop_on_close = True)

    def generate_animation(self):
        pass
