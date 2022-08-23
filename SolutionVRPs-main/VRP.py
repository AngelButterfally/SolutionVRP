"""Simple Vehicles Routing Problem (VRP)."""
# from ast import Import
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from data import create_data_VRP


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
                    math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))))
    return distances


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += 'Distance of the route: {} pix\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {} pix'.format(max_route_distance))
    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print('No solution found !')


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        route_distance = 0
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, route_nbr)
        routes.append(route)
    return routes


def VRP_solution(input_data):
    '''
    需要调用的主程序
    '''
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(input_data['locations']),
                                           input_data['num_vehicles'], input_data['depot'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(
        input_data['locations'])
    # Create and register a transit callback.

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        30000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    # Setting first solution heuristic.
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.first_solution_strategy = (
    #     routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    search_parameters.time_limit.seconds = 20
    search_parameters.log_search = True
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        vrp_rout = get_routes(solution, routing, manager)
        return vrp_rout
    else:
        print('No solution found !')


def draw_pic(input_data):
    '''
    可调用的画图程序
    '''
    routes = VRP_solution(input_data=input_data)
    colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
    plt.figure(figsize=(8, 6))
    # 画出depot
    depot_coor = input_data['locations'][input_data['depot']]
    plt.plot(depot_coor[0], depot_coor[1], 'r*', markersize=11)
    # 路径可视化
    for k in range(0, len(routes)):
        for i, j in zip(routes[k], routes[k][1:]):
            start_coor = input_data['locations'][i]
            end_coor = input_data['locations'][j]
            plt.arrow(start_coor[0], start_coor[1], end_coor[0] - start_coor[0], end_coor[1] -
                      start_coor[1], head_width=20, ec=mcolors.TABLEAU_COLORS[colors[k]])
    print(input_data['locations'])
    print(routes)
    plt.xlabel("X coordinate", fontsize=14)
    plt.ylabel("Y coordinate", fontsize=14)
    plt.title("VRP path for BIT with vehicle Num: {},receive num:{}".format(
        input_data['num_vehicles'], input_data['receive_point_num']), fontsize=16)
    plt.show()


if __name__ == '__main__':
    #draw_pic(4, replace_point_num=20)
    # a = VRP_solution(1, 42)
    # print(a)
    data = create_data_VRP(
        './point.csv', vehicles_num=3, random_point_num=30, auto_random_select_point=True)
    draw_pic(data)
    # generate_data('./custumer.csv')
    # print(a)
