# =============================================================================
# 24F0633 - MODULE 1: Intelligent Urban Delivery Robot.
# =============================================================================
# This script simulates an intelligent delivery robot in a 15x15 urban grid.
# It builds roads, buildings/houses, and traffic zones; computes delivery
# routes using multiple search algorithms; animates robot movement; and prints
# algorithm performance metrics.

import argparse
import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Set, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# Global Configuration
# =============================================================================
GRID_SIZE = 15
DELIVERY_COUNT = 5
OBSTACLE_RATIO = 0.20
TRAFFIC_RATIO = 0.15
ROAD_COST_RANGE = (1, 5)
TRAFFIC_COST_RANGE = (10, 20)
BASE_STATION = (0, 0)
RANDOM_SEED = 42
ANIMATION_DELAY_SECONDS = 0.20
DEFAULT_VISUAL_ALGORITHM = "A* Manhattan"
MAX_SCENARIO_ATTEMPTS = 200

Coordinate = Tuple[int, int]


class EnvironmentData(TypedDict):
    """Stores all environment components for the urban grid."""

    size: int
    obstacles: Set[Coordinate]
    traffic: Set[Coordinate]
    costs: Dict[Coordinate, int]


SearchFunction = Callable[
    [Coordinate, Coordinate, EnvironmentData], Tuple[Optional[List[Coordinate]], int]
]


@dataclass
class DeliveryStepResult:
    """Represents one delivery attempt for one search algorithm."""

    delivery_number: int
    start_cell: Coordinate
    goal_cell: Coordinate
    status: str
    path: Optional[List[Coordinate]]
    path_cost: int
    nodes_explored: int
    execution_time_seconds: float


@dataclass
class AlgorithmPerformance:
    """Stores aggregated performance for one algorithm over all deliveries."""

    algorithm_name: str
    completed_deliveries: int = 0
    total_cost: int = 0
    total_nodes_explored: int = 0
    total_execution_time_seconds: float = 0.0
    delivery_results: List[DeliveryStepResult] = field(default_factory=list)


# =============================================================================
# QUESTION 1: ENVIRONMENT REPRESENTATION
# Brief Description:
# This section creates the 15x15 environment, places buildings/houses and
# traffic zones, assigns traversal costs, and generates reachable deliveries.
# =============================================================================


def generate_unique_cells(
    grid_size: int,
    number_of_cells: int,
    forbidden_cells: Set[Coordinate],
) -> List[Coordinate]:
    """Return unique random cells while excluding forbidden cells."""
    available_cells = [
        (row_index, column_index)
        for row_index in range(grid_size)
        for column_index in range(grid_size)
        if (row_index, column_index) not in forbidden_cells
    ]

    if number_of_cells > len(available_cells):
        raise ValueError(
            "Cannot sample requested cells because available cells are fewer."
        )

    return random.sample(available_cells, number_of_cells)


def create_environment(
    grid_size: int,
    obstacle_ratio: float,
    traffic_ratio: float,
    base_station: Coordinate,
) -> EnvironmentData:
    """Create city environment with obstacles, traffic zones, and cell costs."""
    total_cells = grid_size * grid_size
    obstacle_count = round(total_cells * obstacle_ratio)
    traffic_count = round(total_cells * traffic_ratio)

    obstacle_cells = set(
        generate_unique_cells(
            grid_size=grid_size,
            number_of_cells=obstacle_count,
            forbidden_cells={base_station},
        )
    )

    traffic_forbidden = set(obstacle_cells) | {base_station}
    max_traffic_cells = total_cells - len(traffic_forbidden)
    traffic_count = min(traffic_count, max_traffic_cells)

    traffic_cells = set(
        generate_unique_cells(
            grid_size=grid_size,
            number_of_cells=traffic_count,
            forbidden_cells=traffic_forbidden,
        )
    )

    traversal_costs: Dict[Coordinate, int] = {}
    for row_index in range(grid_size):
        for column_index in range(grid_size):
            cell = (row_index, column_index)
            if cell in obstacle_cells:
                continue
            if cell in traffic_cells:
                traversal_costs[cell] = random.randint(
                    TRAFFIC_COST_RANGE[0], TRAFFIC_COST_RANGE[1]
                )
            else:
                traversal_costs[cell] = random.randint(
                    ROAD_COST_RANGE[0], ROAD_COST_RANGE[1]
                )

    return EnvironmentData(
        size=grid_size,
        obstacles=obstacle_cells,
        traffic=traffic_cells,
        costs=traversal_costs,
    )


def is_passable(cell: Coordinate, environment: EnvironmentData) -> bool:
    """Return True when a cell is inside bounds and not a building/house."""
    row_index, column_index = cell
    grid_size = environment["size"]
    inside_grid = 0 <= row_index < grid_size and 0 <= column_index < grid_size
    if not inside_grid:
        return False
    return cell not in environment["obstacles"]


def get_neighbors(cell: Coordinate, environment: EnvironmentData) -> List[Coordinate]:
    """Return valid four-direction neighbors for a given cell."""
    row_index, column_index = cell
    candidate_neighbors = [
        (row_index - 1, column_index),
        (row_index + 1, column_index),
        (row_index, column_index - 1),
        (row_index, column_index + 1),
    ]
    return [neighbor for neighbor in candidate_neighbors if is_passable(neighbor, environment)]


def bfs_reachable_cells(
    environment: EnvironmentData,
    start_cell: Coordinate,
) -> Set[Coordinate]:
    """Return all cells reachable from start using BFS traversal."""
    reachable_cells: Set[Coordinate] = set()
    frontier_queue: deque[Coordinate] = deque([start_cell])

    while frontier_queue:
        current_cell = frontier_queue.popleft()
        if current_cell in reachable_cells:
            continue

        reachable_cells.add(current_cell)
        for neighbor_cell in get_neighbors(current_cell, environment):
            if neighbor_cell not in reachable_cells:
                frontier_queue.append(neighbor_cell)

    return reachable_cells


def generate_delivery_locations(
    environment: EnvironmentData,
    base_station: Coordinate,
    delivery_count: int,
) -> List[Coordinate]:
    """Generate random unique deliveries only from reachable cells."""
    reachable_cells = bfs_reachable_cells(environment, base_station)
    candidate_delivery_cells = sorted(reachable_cells - {base_station})

    if len(candidate_delivery_cells) < delivery_count:
        raise ValueError("Not enough reachable cells for requested deliveries.")

    return random.sample(candidate_delivery_cells, delivery_count)


def are_all_deliveries_reachable(
    environment: EnvironmentData,
    base_station: Coordinate,
    delivery_locations: List[Coordinate],
) -> bool:
    """Check whether every delivery location is reachable from base station."""
    reachable_cells = bfs_reachable_cells(environment, base_station)
    return all(delivery_cell in reachable_cells for delivery_cell in delivery_locations)


def generate_valid_scenario() -> Tuple[EnvironmentData, List[Coordinate]]:
    """Create environment and deliveries until all deliveries are reachable."""
    for _ in range(MAX_SCENARIO_ATTEMPTS):
        environment = create_environment(
            grid_size=GRID_SIZE,
            obstacle_ratio=OBSTACLE_RATIO,
            traffic_ratio=TRAFFIC_RATIO,
            base_station=BASE_STATION,
        )

        try:
            delivery_locations = generate_delivery_locations(
                environment=environment,
                base_station=BASE_STATION,
                delivery_count=DELIVERY_COUNT,
            )
        except ValueError:
            continue

        if are_all_deliveries_reachable(
            environment=environment,
            base_station=BASE_STATION,
            delivery_locations=delivery_locations,
        ):
            return environment, delivery_locations

    raise RuntimeError("Unable to generate a valid reachable scenario.")


# =============================================================================
# QUESTION 2: ALGORITHM IMPLEMENTATION
# Brief Description:
# This section implements BFS, DFS, UCS, Greedy Best-First Search, and A*.
# Manhattan and Euclidean heuristics can be toggled through wrappers.
# =============================================================================


def reconstruct_path(
    predecessor_map: Dict[Coordinate, Optional[Coordinate]],
    start_cell: Coordinate,
    goal_cell: Coordinate,
) -> Optional[List[Coordinate]]:
    """Reconstruct path from predecessor map and return start-to-goal list."""
    if goal_cell not in predecessor_map:
        return None

    path: List[Coordinate] = []
    current_cell: Optional[Coordinate] = goal_cell
    while current_cell is not None:
        path.append(current_cell)
        current_cell = predecessor_map[current_cell]

    path.reverse()
    if path and path[0] == start_cell:
        return path
    return None


def calculate_path_cost(
    path: Optional[List[Coordinate]],
    environment: EnvironmentData,
) -> int:
    """Calculate full traversal cost of a path excluding starting cell."""
    if not path or len(path) <= 1:
        return 0
    return sum(environment["costs"][cell] for cell in path[1:])


def manhattan_distance(cell_a: Coordinate, cell_b: Coordinate) -> float:
    """Return Manhattan distance between two cells."""
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])


def euclidean_distance(cell_a: Coordinate, cell_b: Coordinate) -> float:
    """Return Euclidean distance between two cells."""
    row_distance = cell_a[0] - cell_b[0]
    column_distance = cell_a[1] - cell_b[1]
    return math.sqrt(row_distance**2 + column_distance**2)


def get_heuristic_function(
    heuristic_name: str,
) -> Callable[[Coordinate, Coordinate], float]:
    """Return heuristic function according to selected heuristic name."""
    heuristic_key = heuristic_name.strip().lower()
    if heuristic_key == "manhattan":
        return manhattan_distance
    if heuristic_key == "euclidean":
        return euclidean_distance
    raise ValueError("Heuristic must be either 'Manhattan' or 'Euclidean'.")


def breadth_first_search(
    start_cell: Coordinate,
    goal_cell: Coordinate,
    environment: EnvironmentData,
) -> Tuple[Optional[List[Coordinate]], int]:
    """Find path using Breadth First Search and return path with nodes explored."""
    frontier_queue: deque[Coordinate] = deque([start_cell])
    predecessor_map: Dict[Coordinate, Optional[Coordinate]] = {start_cell: None}
    nodes_explored = 0

    while frontier_queue:
        current_cell = frontier_queue.popleft()
        nodes_explored += 1

        if current_cell == goal_cell:
            break

        for neighbor_cell in get_neighbors(current_cell, environment):
            if neighbor_cell not in predecessor_map:
                predecessor_map[neighbor_cell] = current_cell
                frontier_queue.append(neighbor_cell)

    return reconstruct_path(predecessor_map, start_cell, goal_cell), nodes_explored


def depth_first_search(
    start_cell: Coordinate,
    goal_cell: Coordinate,
    environment: EnvironmentData,
) -> Tuple[Optional[List[Coordinate]], int]:
    """Find path using Depth First Search and return path with nodes explored."""
    frontier_stack: List[Coordinate] = [start_cell]
    predecessor_map: Dict[Coordinate, Optional[Coordinate]] = {start_cell: None}
    nodes_explored = 0

    while frontier_stack:
        current_cell = frontier_stack.pop()
        nodes_explored += 1

        if current_cell == goal_cell:
            break

        for neighbor_cell in reversed(get_neighbors(current_cell, environment)):
            if neighbor_cell not in predecessor_map:
                predecessor_map[neighbor_cell] = current_cell
                frontier_stack.append(neighbor_cell)

    return reconstruct_path(predecessor_map, start_cell, goal_cell), nodes_explored


def uniform_cost_search(
    start_cell: Coordinate,
    goal_cell: Coordinate,
    environment: EnvironmentData,
) -> Tuple[Optional[List[Coordinate]], int]:
    """Find minimum-cost path using Uniform Cost Search."""
    tie_breaker = count()
    priority_queue: List[Tuple[int, int, Coordinate]] = []
    heapq.heappush(priority_queue, (0, next(tie_breaker), start_cell))

    predecessor_map: Dict[Coordinate, Optional[Coordinate]] = {start_cell: None}
    best_cost_to_cell: Dict[Coordinate, int] = {start_cell: 0}
    nodes_explored = 0

    while priority_queue:
        current_cost, _, current_cell = heapq.heappop(priority_queue)

        if current_cost > best_cost_to_cell.get(current_cell, math.inf):
            continue

        nodes_explored += 1
        if current_cell == goal_cell:
            break

        for neighbor_cell in get_neighbors(current_cell, environment):
            step_cost = environment["costs"][neighbor_cell]
            new_cost = current_cost + step_cost

            if new_cost < best_cost_to_cell.get(neighbor_cell, math.inf):
                best_cost_to_cell[neighbor_cell] = new_cost
                predecessor_map[neighbor_cell] = current_cell
                heapq.heappush(
                    priority_queue,
                    (new_cost, next(tie_breaker), neighbor_cell),
                )

    return reconstruct_path(predecessor_map, start_cell, goal_cell), nodes_explored


def greedy_best_first_search(
    start_cell: Coordinate,
    goal_cell: Coordinate,
    environment: EnvironmentData,
    heuristic_name: str,
) -> Tuple[Optional[List[Coordinate]], int]:
    """Find path using Greedy Best-First Search with selected heuristic."""
    heuristic_function = get_heuristic_function(heuristic_name)
    tie_breaker = count()
    priority_queue: List[Tuple[float, int, Coordinate]] = []
    heapq.heappush(
        priority_queue,
        (heuristic_function(start_cell, goal_cell), next(tie_breaker), start_cell),
    )

    predecessor_map: Dict[Coordinate, Optional[Coordinate]] = {start_cell: None}
    visited_cells: Set[Coordinate] = set()
    nodes_explored = 0

    while priority_queue:
        _, _, current_cell = heapq.heappop(priority_queue)
        if current_cell in visited_cells:
            continue

        visited_cells.add(current_cell)
        nodes_explored += 1
        if current_cell == goal_cell:
            break

        for neighbor_cell in get_neighbors(current_cell, environment):
            if neighbor_cell in visited_cells or neighbor_cell in predecessor_map:
                continue

            predecessor_map[neighbor_cell] = current_cell
            neighbor_priority = heuristic_function(neighbor_cell, goal_cell)
            heapq.heappush(
                priority_queue,
                (neighbor_priority, next(tie_breaker), neighbor_cell),
            )

    return reconstruct_path(predecessor_map, start_cell, goal_cell), nodes_explored


def a_star_search(
    start_cell: Coordinate,
    goal_cell: Coordinate,
    environment: EnvironmentData,
    heuristic_name: str,
) -> Tuple[Optional[List[Coordinate]], int]:
    """Find path using A* Search with selected heuristic."""
    heuristic_function = get_heuristic_function(heuristic_name)
    tie_breaker = count()
    priority_queue: List[Tuple[float, int, Coordinate]] = []

    start_priority = heuristic_function(start_cell, goal_cell)
    heapq.heappush(
        priority_queue,
        (start_priority, next(tie_breaker), start_cell),
    )

    predecessor_map: Dict[Coordinate, Optional[Coordinate]] = {start_cell: None}
    best_cost_to_cell: Dict[Coordinate, int] = {start_cell: 0}
    nodes_explored = 0

    while priority_queue:
        current_priority, _, current_cell = heapq.heappop(priority_queue)
        current_best_cost = best_cost_to_cell.get(current_cell, math.inf)
        expected_priority = current_best_cost + heuristic_function(current_cell, goal_cell)

        if current_priority > expected_priority + 1e-9:
            continue

        nodes_explored += 1
        if current_cell == goal_cell:
            break

        for neighbor_cell in get_neighbors(current_cell, environment):
            step_cost = environment["costs"][neighbor_cell]
            tentative_cost = current_best_cost + step_cost

            if tentative_cost < best_cost_to_cell.get(neighbor_cell, math.inf):
                best_cost_to_cell[neighbor_cell] = tentative_cost
                predecessor_map[neighbor_cell] = current_cell
                neighbor_priority = tentative_cost + heuristic_function(
                    neighbor_cell,
                    goal_cell,
                )
                heapq.heappush(
                    priority_queue,
                    (neighbor_priority, next(tie_breaker), neighbor_cell),
                )

    return reconstruct_path(predecessor_map, start_cell, goal_cell), nodes_explored

def create_greedy_wrapper(heuristic_name: str) -> SearchFunction:
    """Create a Greedy search wrapper that toggles the heuristic type."""

    def wrapped(
        start_cell: Coordinate,
        goal_cell: Coordinate,
        environment: EnvironmentData,
    ) -> Tuple[Optional[List[Coordinate]], int]:
        """Run Greedy Best-First Search with fixed heuristic."""
        return greedy_best_first_search(
            start_cell=start_cell,
            goal_cell=goal_cell,
            environment=environment,
            heuristic_name=heuristic_name,
        )

    return wrapped


def create_a_star_wrapper(heuristic_name: str) -> SearchFunction:
    """Create an A* wrapper that toggles the heuristic type."""

    def wrapped(
        start_cell: Coordinate,
        goal_cell: Coordinate,
        environment: EnvironmentData,
    ) -> Tuple[Optional[List[Coordinate]], int]:
        """Run A* Search with fixed heuristic."""
        return a_star_search(
            start_cell=start_cell,
            goal_cell=goal_cell,
            environment=environment,
            heuristic_name=heuristic_name,
        )

    return wrapped


def build_algorithm_map() -> Dict[str, SearchFunction]:
    """Build the complete algorithm registry used for evaluation."""
    return {
        "BFS": breadth_first_search,
        "DFS": depth_first_search,
        "UCS": uniform_cost_search,
        "Greedy Manhattan": create_greedy_wrapper("Manhattan"),
        "Greedy Euclidean": create_greedy_wrapper("Euclidean"),
        "A* Manhattan": create_a_star_wrapper("Manhattan"),
        "A* Euclidean": create_a_star_wrapper("Euclidean"),
    }


# =============================================================================
# QUESTION 3: DELIVERY EXECUTION PROCESS
# Brief Description:
# This section executes sequential deliveries. The robot starts at base station,
# then moves Delivery 1 to Delivery 5, updating start location each time.
# =============================================================================


def execute_delivery_sequence(
    environment: EnvironmentData,
    start_cell: Coordinate,
    delivery_locations: List[Coordinate],
    search_algorithm: SearchFunction,
    algorithm_name: str,
) -> AlgorithmPerformance:
    """Run one algorithm over all deliveries in sequence and collect metrics."""
    performance = AlgorithmPerformance(algorithm_name=algorithm_name)
    current_start = start_cell

    for delivery_index, delivery_goal in enumerate(delivery_locations, start=1):
        start_time = time.perf_counter()
        path, nodes_explored = search_algorithm(
            current_start,
            delivery_goal,
            environment,
        )
        elapsed_time = time.perf_counter() - start_time

        if path is None:
            performance.delivery_results.append(
                DeliveryStepResult(
                    delivery_number=delivery_index,
                    start_cell=current_start,
                    goal_cell=delivery_goal,
                    status="No Path Found",
                    path=None,
                    path_cost=0,
                    nodes_explored=nodes_explored,
                    execution_time_seconds=elapsed_time,
                )
            )
            performance.total_nodes_explored += nodes_explored
            performance.total_execution_time_seconds += elapsed_time
            break

        path_cost = calculate_path_cost(path, environment)
        performance.delivery_results.append(
            DeliveryStepResult(
                delivery_number=delivery_index,
                start_cell=current_start,
                goal_cell=delivery_goal,
                status="Delivered",
                path=path,
                path_cost=path_cost,
                nodes_explored=nodes_explored,
                execution_time_seconds=elapsed_time,
            )
        )

        performance.completed_deliveries += 1
        performance.total_cost += path_cost
        performance.total_nodes_explored += nodes_explored
        performance.total_execution_time_seconds += elapsed_time

        current_start = delivery_goal

    return performance


def evaluate_all_algorithms(
    environment: EnvironmentData,
    base_station: Coordinate,
    delivery_locations: List[Coordinate],
    algorithm_map: Dict[str, SearchFunction],
) -> Dict[str, AlgorithmPerformance]:
    """Evaluate all algorithms and return a performance report dictionary."""
    performance_report: Dict[str, AlgorithmPerformance] = {}

    for algorithm_name, search_algorithm in algorithm_map.items():
        performance_report[algorithm_name] = execute_delivery_sequence(
            environment=environment,
            start_cell=base_station,
            delivery_locations=delivery_locations,
            search_algorithm=search_algorithm,
            algorithm_name=algorithm_name,
        )

    return performance_report


# =============================================================================
# QUESTION 5: VISUALIZATION (MATPLOTLIB)
# Brief Description:
# This section animates robot movement cell-by-cell using matplotlib. The legend
# uses required labels exactly: Buildings/Houses, City Streets, Traffic Zones,
# Base Station, and Robot.
# =============================================================================


def build_display_grid(
    environment: EnvironmentData,
    base_station: Coordinate,
) -> List[List[int]]:
    """Create numeric grid used by matplotlib for background rendering."""
    grid_size = environment["size"]
    display_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    for obstacle_cell in environment["obstacles"]:
        display_grid[obstacle_cell[0]][obstacle_cell[1]] = 1

    for traffic_cell in environment["traffic"]:
        display_grid[traffic_cell[0]][traffic_cell[1]] = 2

    base_row, base_column = base_station
    display_grid[base_row][base_column] = 3
    return display_grid


def setup_visualization(
    environment: EnvironmentData,
    base_station: Coordinate,
    delivery_locations: List[Coordinate],
    algorithm_name: str,
):
    """Initialize matplotlib figure, map, markers, and required legend labels."""
    plt.ion()
    figure, axis = plt.subplots(figsize=(8, 8))

    color_map = ListedColormap(
        [
            "#d9d9d9",  # City Streets
            "#2f2f2f",  # Buildings/Houses
            "#f4a261",  # Traffic Zones
            "#457b9d",  # Base Station
        ]
    )

    display_grid = build_display_grid(environment, base_station)
    axis.imshow(display_grid, cmap=color_map, vmin=0, vmax=3)
    axis.set_title(f"Intelligent Urban Delivery Robot - {algorithm_name}")
    axis.set_xticks(range(environment["size"]))
    axis.set_yticks(range(environment["size"]))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid(color="white", linewidth=0.5)

    delivery_marker_points = [[cell[1], cell[0]] for cell in delivery_locations]
    delivery_scatter = axis.scatter(
        [point[0] for point in delivery_marker_points],
        [point[1] for point in delivery_marker_points],
        c="#e9c46a",
        marker="*",
        s=120,
        edgecolors="black",
        linewidths=0.5,
    )

    robot_marker, = axis.plot(
        [base_station[1]],
        [base_station[0]],
        marker="o",
        color="red",
        markersize=8,
        linestyle="None",
    )

    path_line, = axis.plot([], [], color="#00b4d8", linewidth=2.0)

    legend_handles = [
        Patch(facecolor="#2f2f2f", edgecolor="black", label="Buildings/Houses"),
        Patch(facecolor="#d9d9d9", edgecolor="black", label="City Streets"),
        Patch(facecolor="#f4a261", edgecolor="black", label="Traffic Zones"),
        Patch(facecolor="#457b9d", edgecolor="black", label="Base Station"),
        Line2D([0], [0], marker="o", color="red", linestyle="None", label="Robot"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=9)

    return figure, axis, robot_marker, path_line, delivery_scatter


def update_delivery_markers(
    delivery_scatter,
    remaining_deliveries: List[Coordinate],
) -> None:
    """Update delivery marker positions by removing completed destinations."""
    points = [[cell[1], cell[0]] for cell in remaining_deliveries]
    if points:
        delivery_scatter.set_offsets(points)
    else:
        delivery_scatter.set_offsets(np.empty((0, 2)))


def animate_robot_along_path(
    figure,
    axis,
    robot_marker,
    path_line,
    path: List[Coordinate],
    algorithm_name: str,
    delivery_number: int,
) -> None:
    """Animate robot cell-by-cell movement for one delivery path."""
    traveled_x: List[int] = []
    traveled_y: List[int] = []

    for step_index, step_cell in enumerate(path):
        traveled_x.append(step_cell[1])
        traveled_y.append(step_cell[0])

        robot_marker.set_data([step_cell[1]], [step_cell[0]])
        path_line.set_data(traveled_x, traveled_y)
        axis.set_title(
            f"{algorithm_name} | Delivery {delivery_number}/{DELIVERY_COUNT} | "
            f"Step {step_index}/{len(path) - 1}"
        )

        figure.canvas.draw_idle()
        plt.pause(ANIMATION_DELAY_SECONDS)


def simulate_delivery_animation(
    environment: EnvironmentData,
    base_station: Coordinate,
    delivery_locations: List[Coordinate],
    algorithm_name: str,
    algorithm_function: SearchFunction,
) -> None:
    """Run animated simulation for one selected algorithm."""
    figure, axis, robot_marker, path_line, delivery_scatter = setup_visualization(
        environment=environment,
        base_station=base_station,
        delivery_locations=delivery_locations,
        algorithm_name=algorithm_name,
    )

    remaining_deliveries = list(delivery_locations)
    current_position = base_station

    for delivery_number, delivery_cell in enumerate(delivery_locations, start=1):
        path, _ = algorithm_function(current_position, delivery_cell, environment)
        if path is None:
            print(
                f"Visualization stopped: no path from {current_position} "
                f"to {delivery_cell}."
            )
            break

        animate_robot_along_path(
            figure=figure,
            axis=axis,
            robot_marker=robot_marker,
            path_line=path_line,
            path=path,
            algorithm_name=algorithm_name,
            delivery_number=delivery_number,
        )

        current_position = delivery_cell
        if delivery_cell in remaining_deliveries:
            remaining_deliveries.remove(delivery_cell)
        update_delivery_markers(delivery_scatter, remaining_deliveries)
        figure.canvas.draw_idle()
        plt.pause(ANIMATION_DELAY_SECONDS)

    plt.ioff()
    plt.show()

# =============================================================================
# QUESTION 4: PERFORMANCE EVALUATION
# Brief Description:
# This section prints the required final performance table after simulation with
# columns: Algorithm Name, Completed Deliveries, Total Cost, Nodes Explored,
# Execution Time, and Average Path Cost (Total Cost / 5).
# =============================================================================


def print_environment_summary(
    environment: EnvironmentData,
    delivery_locations: List[Coordinate],
) -> None:
    """Print basic environment details and generated delivery assignments."""
    print("\nEnvironment Summary")
    print("-" * 70)
    print(f"Grid Size: {environment['size']} x {environment['size']}")
    print(f"Buildings/Houses: {len(environment['obstacles'])}")
    print(f"Traffic Zones: {len(environment['traffic'])}")
    print(f"Base Station: {BASE_STATION}")
    print(f"Delivery Locations: {delivery_locations}")


def print_performance_table(
    performance_report: Dict[str, AlgorithmPerformance],
) -> None:
    """Print final required performance table with all mandatory columns."""
    print("\nPerformance Table")
    print("-" * 132)
    header = (
        f"{'Algorithm Name':25} | {'Completed Deliveries':21} | {'Total Cost':10} | "
        f"{'Nodes Explored':14} | {'Execution Time':14} | "
        f"{'Average Path Cost (Total Cost / 5)':32}"
    )
    print(header)
    print("-" * 132)

    for algorithm_name, performance in performance_report.items():
        completed_text = f"{performance.completed_deliveries}/{DELIVERY_COUNT}"
        average_path_cost = performance.total_cost / DELIVERY_COUNT

        print(
            f"{algorithm_name:25} | {completed_text:21} | {performance.total_cost:10} | "
            f"{performance.total_nodes_explored:14} | "
            f"{performance.total_execution_time_seconds:14.6f} | "
            f"{average_path_cost:32.2f}"
        )


def parse_runtime_options() -> argparse.Namespace:
    """Parse optional runtime controls without changing strict project rules."""
    parser = argparse.ArgumentParser(
        description="24F0633 - MODULE 1 Intelligent Urban Delivery Robot"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable matplotlib animation during script execution.",
    )
    parser.add_argument(
        "--visual-algorithm",
        type=str,
        default=DEFAULT_VISUAL_ALGORITHM,
        help="Algorithm to use for animation only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run complete simulation: setup, evaluation, animation, and reporting."""
    options = parse_runtime_options()
    random.seed(RANDOM_SEED)

    print("Robot waiting at base station for assignment...")
    time.sleep(1)

    environment, delivery_locations = generate_valid_scenario()
    print_environment_summary(environment, delivery_locations)

    if not are_all_deliveries_reachable(
        environment=environment,
        base_station=BASE_STATION,
        delivery_locations=delivery_locations,
    ):
        raise RuntimeError("Delivery locations are not reachable from base station.")

    algorithm_map = build_algorithm_map()
    if options.visual_algorithm not in algorithm_map:
        valid_algorithms = ", ".join(algorithm_map.keys())
        raise ValueError(
            f"Invalid visual algorithm '{options.visual_algorithm}'. "
            f"Choose one of: {valid_algorithms}"
        )

    performance_report = evaluate_all_algorithms(
        environment=environment,
        base_station=BASE_STATION,
        delivery_locations=delivery_locations,
        algorithm_map=algorithm_map,
    )

    if not options.no_visualization:
        simulate_delivery_animation(
            environment=environment,
            base_station=BASE_STATION,
            delivery_locations=delivery_locations,
            algorithm_name=options.visual_algorithm,
            algorithm_function=algorithm_map[options.visual_algorithm],
        )
    else:
        print("\nVisualization skipped by --no-visualization.")

    print_performance_table(performance_report)


if __name__ == "__main__":
    main()
