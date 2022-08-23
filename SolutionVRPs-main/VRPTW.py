"""Vehicles Routing Problem (VRP) with Time Windows."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import numpy as np
import pandas as pd

def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (int(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1]))))
    return distances


def compute_time_matrex(distances,speed = 100):
    vehecle_times = {}
    for i in range(0,len(distances)):
        vehecle_times[i] = {}
        for j in range(0,len(distances)):
            vehecle_times[i][j] = int(distances[i][j]/speed)
    return vehecle_times

def create_data_model(num_vehecles = 3,speed = 100):
    """Stores the data for the problem."""
    data = {}

    data['vehecle_speed'] = speed

    data['locations'] =[
    (691, 1233), (255, 804), (518, 695), (521, 770), (688, 661), (691, 1717), (518, 1554), (518, 1715), (573, 1824), (824, 648), (743, 774), (956, 612), (961, 795), 
    (951, 1233), (951, 1615),(875, 1824),(1042, 736),  (1122, 927), 
    (1098, 418), (1174, 608), (1167, 827), (1156, 1119), (1252, 1596), (1117, 1824), (1319, 451),
    (1319, 608), (1536, 840), (1627, 608), (1626, 732), (1672, 840), (1733, 927), (1369, 1824), (1850, 1119), (2089, 998), (1937, 1187), (2176, 1082), (2406, 1373),
    (2021, 1862), (2262, 1740), (2436, 1873), (2426, 1631), (2570, 1093)
    ]
    data['distance'] = compute_euclidean_distance_matrix(data['locations'])
    data['time_matrix'] = compute_time_matrex(data['distance'],speed=speed)

    data['time_windows'] = [(0, 1000), (0, 974), (0, 972), (0, 967), (678, 801), (0, 969),
     (415, 514), (0, 968), (404, 481), (400, 497), (577, 632), (206, 325), (0, 975), 
     (690, 827), (0, 957), (175, 300), (0, 960), (733, 870), (0, 974), (0, 957), (0, 958),
      (0, 971), (409, 494), (206, 325), (0, 960), (817, 956), (0, 978), (104, 255), 
      (0, 983), (0, 960), (259, 354), (0, 972), (0, 956), (45, 200), (0, 953), (686, 813), 
      (41, 208), (0, 968), (302, 405), (33, 224), (360, 437), (396, 511)]

    data['num_vehicles'] = num_vehecles

    data['depot'] = 0
    
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), solution.Min(time_var),
                solution.Max(time_var))
            index = solution.Value(routing.NextVar(index))
        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    solution.Min(time_var),
                                                    solution.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            solution.Min(time_var))
        print(plan_output)
        total_time += solution.Min(time_var)
    print('Total time of all routes: {}min'.format(total_time))

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    distance = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, route_nbr)
        routes.append(route)
        distance.append(route_distance)
    return routes

    routes = get_routes(solution, routing, manager)
    # print(routes[0])
    # print(routes[1])
    print(routes[0],routes[1])

def VRPTW_solution(num = 3):
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
    data = create_data_model(num_vehecles= num,speed=100)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        2000,  # allow waiting time
        2000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        routes = get_routes(solution, routing, manager)
        return routes

