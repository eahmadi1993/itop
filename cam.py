from itertools import accumulate
from typing import List
import uuid
import numpy as np
from routes import ThetaFinder, IntersectionLayout
from sim_mpc import MPCC, MPCCParams
from system import LinearSystem, NonlinearSystem
import drawnow
import matplotlib.pyplot as plt


class Vehicle:
    def __init__(self, initial_condition, orientation):
        self.id = str(uuid.uuid4())
        self.initial_condition = initial_condition
        self.orientation = orientation
        self.current_states = None
        self.current_progress = None
        self.state_predictions = None
        self.vir_predictions = None
        self.input_predictions = None
        self.progress_predictions = None
        self.traj = None
        self.x_trajectory = []  # vehicle x position from initial condition to current time.
        self.y_trajectory = []  # vehicle y position from initial condition to current time.
        self.x_trajectory.append(initial_condition[0])
        self.y_trajectory.append(initial_condition[1])


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
    def __init__(self, mpcc: MPCC):
        self.mpcc = mpcc
        self.sim_steps = None
        self.arrivals = 3  # subject to change
        self.flow = 100  # subject to change
        self.rng = np.random.default_rng()
        self.west_arrival = None
        self.south_arrival = None
        self.east_arrival = None
        self.north_arrival = None
        self.additional_steps = 500
        self.set_arrivals()

        self.vehicle_manager = VehicleManager()

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

    def update_vehicle_states(self):
        sys_nl = NonlinearSystem(self.mpcc.sys.dt, self.mpcc.sys.model.lr, self.mpcc.sys.model.lf)
        for veh in self.vehicle_manager.west_vehicles_list:
            veh.current_states = sys_nl.update_nls_states(veh.current_states,
                                                          veh.input_predictions[:, 0].reshape(-1, 1))
            self.mpcc.theta_finder.set_initial_conditions(veh.initial_condition[0], veh.initial_condition[1],
                                                          path = veh.traj.name)
            veh.current_progress = self.mpcc.theta_finder.find_theta(veh.current_states[0], veh.current_states[1])
            veh.current_states = self.mpcc.unwrap_x0(veh.current_states)
            veh.x_trajectory.append(veh.current_states[0])
            veh.y_trajectory.append(veh.current_states[1])

        for veh in self.vehicle_manager.east_vehicles_list:
            veh.current_states = sys_nl.update_nls_states(veh.current_states,
                                                          veh.input_predictions[:, 0].reshape(-1, 1))
            self.mpcc.theta_finder.set_initial_conditions(veh.initial_condition[0], veh.initial_condition[1],
                                                          path = veh.traj.name)
            veh.current_progress = self.mpcc.theta_finder.find_theta(veh.current_states[0], veh.current_states[1])
            veh.current_states = self.mpcc.unwrap_x0(veh.current_states)
            veh.x_trajectory.append(veh.current_states[0])
            veh.y_trajectory.append(veh.current_states[1])

        for veh in self.vehicle_manager.north_vehicles_list:
            veh.current_states = sys_nl.update_nls_states(veh.current_states,
                                                          veh.input_predictions[:, 0].reshape(-1, 1))
            self.mpcc.theta_finder.set_initial_conditions(veh.initial_condition[0], veh.initial_condition[1],
                                                          path = veh.traj.name)
            veh.current_progress = self.mpcc.theta_finder.find_theta(veh.current_states[0], veh.current_states[1])
            veh.current_states = self.mpcc.unwrap_x0(veh.current_states)
            veh.x_trajectory.append(veh.current_states[0])
            veh.y_trajectory.append(veh.current_states[1])

        for veh in self.vehicle_manager.south_vehicles_list:
            veh.current_states = sys_nl.update_nls_states(veh.current_states,
                                                          veh.input_predictions[:, 0].reshape(-1, 1))
            self.mpcc.theta_finder.set_initial_conditions(veh.initial_condition[0], veh.initial_condition[1],
                                                          path = veh.traj.name)
            veh.current_progress = self.mpcc.theta_finder.find_theta(veh.current_states[0], veh.current_states[1])
            veh.current_states = self.mpcc.unwrap_x0(veh.current_states)
            veh.x_trajectory.append(veh.current_states[0])
            veh.y_trajectory.append(veh.current_states[1])

    def shift_predictions(self):
        for veh in self.vehicle_manager.west_vehicles_list:
            veh.state_predictions, veh.progress_predictions, veh.input_predictions, veh.vir_predictions = \
                self.mpcc.get_shift_prediction(veh.state_predictions,
                                               veh.progress_predictions,
                                               veh.input_predictions,
                                               veh.vir_predictions,
                                               veh.current_states,
                                               veh.current_progress)

        for veh in self.vehicle_manager.east_vehicles_list:
            veh.state_predictions, veh.progress_predictions, veh.input_predictions, veh.vir_predictions = \
                self.mpcc.get_shift_prediction(veh.state_predictions,
                                               veh.progress_predictions,
                                               veh.input_predictions,
                                               veh.vir_predictions,
                                               veh.current_states,
                                               veh.current_progress)

        for veh in self.vehicle_manager.north_vehicles_list:
            veh.state_predictions, veh.progress_predictions, veh.input_predictions, veh.vir_predictions = \
                self.mpcc.get_shift_prediction(veh.state_predictions,
                                               veh.progress_predictions,
                                               veh.input_predictions,
                                               veh.vir_predictions,
                                               veh.current_states,
                                               veh.current_progress)

        for veh in self.vehicle_manager.south_vehicles_list:
            veh.state_predictions, veh.progress_predictions, veh.input_predictions, veh.vir_predictions = \
                self.mpcc.get_shift_prediction(veh.state_predictions,
                                               veh.progress_predictions,
                                               veh.input_predictions,
                                               veh.vir_predictions,
                                               veh.current_states,
                                               veh.current_progress)

    def get_vehicle(self, x0, orientation):
        self.mpcc.theta_finder.set_initial_conditions(x0[0], x0[1], path = None)
        traj = self.mpcc.theta_finder.mytraj
        theta0 = self.mpcc.theta_finder.find_theta(x0[0], x0[1])

        veh = Vehicle(x0, orientation)
        veh.state_predictions, veh.progress_predictions = self.mpcc.get_prediction(x0, theta0, traj)
        veh.input_predictions = np.zeros((self.mpcc.sys.m, self.mpcc.params.N))
        veh.vir_predictions = np.zeros((self.mpcc.params.N, 1))
        veh.current_progress = theta0
        veh.current_states = x0
        veh.traj = traj
        self.display_vehicle_status(veh.traj.name)
        return veh

    def handle_north_approach(self):
        random_point = self.rng.uniform(23, 32.1)
        x0 = np.array([random_point, 60, -np.pi / 2, self.rng.normal(2)], dtype = float).reshape(-1, 1)
        return self.get_vehicle(x0, "north")

    def handle_south_approach(self):
        random_point = self.rng.uniform(32.1, 41)
        x0 = np.array([random_point, 0, np.pi / 2, self.rng.normal(2)], dtype = float).reshape(-1, 1)
        return self.get_vehicle(x0, "south")

    def handle_east_approach(self):
        random_point = self.rng.uniform(29.2, 41)
        x0 = np.array([60, random_point, np.pi, self.rng.normal(2)], dtype = float).reshape(-1, 1)
        return self.get_vehicle(x0, "east")

    def handle_west_approach(self):
        random_point = self.rng.uniform(20, 29)
        x0 = np.array([0, random_point, 0, self.rng.normal(2)], dtype = float).reshape(-1, 1)
        return self.get_vehicle(x0, "west")

    def handle_north_departure(self):
        for veh in self.vehicle_manager.north_vehicles_list:
            if veh.current_states[1] < 0 or veh.current_states[0] > 65 or veh.current_states[0] < 0:
                self.vehicle_manager.remove_veh_north(veh)
                print("vehicle removed", len(self.vehicle_manager.north_vehicles_list))

    def handle_south_departure(self):
        for veh in self.vehicle_manager.south_vehicles_list:
            if veh.current_states[1] > 65 or veh.current_states[0] > 65 or veh.current_states[0] < 0:
                self.vehicle_manager.remove_veh_south(veh)
                print("vehicle removed", len(self.vehicle_manager.south_vehicles_list))

    def handle_east_departure(self):
        for veh in self.vehicle_manager.east_vehicles_list:
            if veh.current_states[0] < 0 or veh.current_states[1] > 65 or veh.current_states[1] < 0:
                self.vehicle_manager.remove_veh_east(veh)
                print("vehicle removed", len(self.vehicle_manager.east_vehicles_list))

    def handle_west_departure(self):
        for veh in self.vehicle_manager.west_vehicles_list:
            if veh.current_states[0] > 65 or veh.current_states[1] > 65 or veh.current_states[1] < 0:
                self.vehicle_manager.remove_veh_west(veh)
                print("vehicle removed", len(self.vehicle_manager.west_vehicles_list))

    def display_vehicle_status(self, route):
        print(f"vehicle approaching -> route: {route}")

    def run_simulation(self):
        intersection = IntersectionLayout(self.mpcc.theta_finder.track, self.mpcc.theta_finder.track.lane_width,
                                          150)

        def draw_fig():
            intersection.plot_intersection()
            for vehicle in self.vehicle_manager.west_vehicles_list:
                plt.plot(vehicle.current_states[0], vehicle.current_states[1], "o")
                plt.plot(vehicle.state_predictions[0, :], vehicle.state_predictions[1, :])
            for vehicle in self.vehicle_manager.east_vehicles_list:
                plt.plot(vehicle.current_states[0], vehicle.current_states[1], "o")
                plt.plot(vehicle.state_predictions[0, :], vehicle.state_predictions[1, :])
            for vehicle in self.vehicle_manager.north_vehicles_list:
                plt.plot(vehicle.current_states[0], vehicle.current_states[1], "o")
                plt.plot(vehicle.state_predictions[0, :], vehicle.state_predictions[1, :])
            for vehicle in self.vehicle_manager.south_vehicles_list:
                plt.plot(vehicle.current_states[0], vehicle.current_states[1], "o")
                plt.plot(vehicle.state_predictions[0, :], vehicle.state_predictions[1, :])
            # for vehicle in self.vehicle_manager.west_vehicles_list:
            #     plt.plot(vehicle.x_trajectory, vehicle.y_trajectory)
            # for vehicle in self.vehicle_manager.east_vehicles_list:
            #     plt.plot(vehicle.x_trajectory, vehicle.y_trajectory)
            # for vehicle in self.vehicle_manager.north_vehicles_list:
            #     plt.plot(vehicle.x_trajectory, vehicle.y_trajectory)
            # for vehicle in self.vehicle_manager.south_vehicles_list:
            #     plt.plot(vehicle.x_trajectory, vehicle.y_trajectory)

        for i in range(self.sim_steps):

            try:
                if self.north_arrival[i] == 1:
                    veh = self.handle_north_approach()
                    self.vehicle_manager.add_veh_north(veh)
                if self.south_arrival[i] == 1:
                    veh = self.handle_south_approach()
                    self.vehicle_manager.add_veh_south(veh)
                if self.east_arrival[i] == 1:
                    veh = self.handle_east_approach()
                    self.vehicle_manager.add_veh_east(veh)
                if self.west_arrival[i] == 1:
                    veh = self.handle_west_approach()
                    self.vehicle_manager.add_veh_west(veh)


            except Exception as exp:
                pass

            if self.vehicle_manager.intersection_is_empty():
                continue

            self.vehicle_manager = self.mpcc.optimize(self.vehicle_manager)
            self.update_vehicle_states()
            self.shift_predictions()

            drawnow.drawnow(draw_fig, stop_on_close = True)

            self.handle_north_departure()
            self.handle_south_departure()
            self.handle_east_departure()
            self.handle_west_departure()
