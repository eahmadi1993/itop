from itertools import accumulate
from typing import List
import uuid
import numpy as np
from routes import ThetaFinder
from sim_mpc import MPCC, MPCCParams
from system import LinearSystem


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

    def get_current_x_init(self):
        if not self.intersection_is_empty():

            x_init = []
            for veh in self.west_vehicles_list:
                x_init.append(veh.initial_condition)
            for veh in self.east_vehicles_list:
                x_init.append(veh.initial_condition)
            for veh in self.north_vehicles_list:
                x_init.append(veh.initial_condition)
            for veh in self.south_vehicles_list:
                x_init.append(veh.initial_condition)
            return x_init
        raise ValueError("Intersection is empty")


class Simulator:
    def __init__(self, params: MPCCParams, system: LinearSystem, theta_finder: ThetaFinder):
        # todo: manage the parameters and the constructor

        self.mpcc = MPCC(params, system, theta_finder)
        self.sim_steps = None
        self.arrivals = 8  # subject to change
        self.flow = 1800  # subject to change
        self.rng = np.random.default_rng()
        self.west_arrival = None
        self.south_arrival = None
        self.east_arrival = None
        self.north_arrival = None
        self.additional_steps = 100
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

    def run_simulation(self):
        for i in range(self.sim_steps):
            if self.north_arrival[i] == 1:
                x = np.array([self.rng.uniform(23, 32.1), 60, -np.pi / 2, self.rng.normal(2)], dtype = float).reshape(
                    -1,
                    1)
                self.vehicle_manager.add_veh_north(Vehicle(
                    x
                ))
            if self.south_arrival[i] == 1:
                x = np.array([self.rng.uniform(32.1, 41), 0, np.pi / 2, self.rng.normal(2)], dtype = float).reshape(-1,
                                                                                                                    1)
                self.vehicle_manager.add_veh_south(Vehicle(
                    x
                ))
            if self.east_arrival[i] == 1:
                x = np.array([60, self.rng.uniform(29.2, 41), np.pi, self.rng.normal(2)], dtype = float).reshape(-1,
                                                                                                                 1)
                self.vehicle_manager.add_veh_east(Vehicle(
                    x
                ))
            if self.west_arrival[i] == 1:
                x = np.array([0, self.rng.uniform(20, 29), 0, self.rng.normal(2)], dtype = float).reshape(-1,
                                                                                                          1)
                self.vehicle_manager.add_veh_west(Vehicle(
                    x
                ))
            if self.vehicle_manager.intersection_is_empty():
                continue

            x_init = self.vehicle_manager.get_current_x_init()
            self.mpcc.set_vehicle_initial_conditions(x_init)
            self.mpcc.run()

    def generate_animation(self):
        pass
