import networkx as nx
from typing import List, Optional, Dict, Set, Tuple
import math
import pickle
import heapq

import numpy as np


class MEAStar:
    """
    Modified Expanding A* (MEA*) pathfinding algorithm.
    Uses Euclidean distance as the heuristic function.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initialize MEA* algorithm with a graph.

        Args:
            graph: NetworkX graph with node coordinates (x, y, z)
        """
        self.graph = graph
        self.heuristic_values = {}
        self.open_list = set()
        self.close_list = set()
        self.parent_list = {}
        self.goal_cost = {}
        self.final_cost = {}

    def calculate_euclidean_distance(self, node1, node2) -> Optional[float]:
        """
        Calculate Euclidean distance between two nodes using their coordinates.

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Distance in cm or None if nodes don't exist
        """
        if node1 not in self.graph.nodes() or node2 not in self.graph.nodes():
            return None

        x1, y1, z1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y'], self.graph.nodes[node1]['z']
        x2, y2, z2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y'], self.graph.nodes[node2]['z']

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        return distance

    def calculate_heuristic_values(self, goal):
        """
        Pre-calculate heuristic values for all nodes to the goal.
        Uses Euclidean distance if coordinates are available.

        Args:
            goal: Goal node ID

        Raises:
            ValueError: If nodes are missing coordinate attributes (x, y, z)
        """
        self.heuristic_values = {}

        # Check if goal node has coordinates
        if 'x' not in self.graph.nodes[goal] or 'y' not in self.graph.nodes[goal] or 'z' not in self.graph.nodes[goal]:
            raise ValueError(
                f"Goal node {goal} is missing coordinate attributes (x, y, z). Cannot calculate heuristic values.")

        for node in self.graph.nodes():
            # Check if current node has coordinates
            if 'x' not in self.graph.nodes[node] or 'y' not in self.graph.nodes[node] or 'z' not in self.graph.nodes[
                node]:
                raise ValueError(
                    f"Node {node} is missing coordinate attributes (x, y, z). Cannot calculate heuristic values.")

            self.heuristic_values[node] = self.calculate_euclidean_distance(node, goal)

    def get_edge_weight(self, node1, node2) -> float:
        """
        Get the edge weight between two nodes.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Edge weight (reads 'cost' attribute from edges)
        """
        if self.graph.has_edge(node1, node2):
            # Read 'cost' attribute from edge
            edge_data = self.graph[node1][node2]
            if 'cost' in edge_data:
                return edge_data['cost']
            else:
                raise ValueError(f"Edge between {node1} and {node2} is missing 'cost' attribute")
        return float('inf')

    def get_neighbors(self, node, goal) -> List:
        """
        Get all neighbors of a node, excluding nodes 0 and 100 unless they are the goal.

        Args:
            node: Node ID
            goal: Goal node ID

        Returns:
            List of neighbor nodes (filtered)
        """
        neighbors = list(self.graph.neighbors(node))

        # Filter out nodes 0 and 100 unless they are the goal
        filtered_neighbors = []
        for neighbor in neighbors:
            if neighbor in [0, 100] and neighbor != goal:
                continue
            filtered_neighbors.append(neighbor)

        return filtered_neighbors

    def backtrack_path(self, goal) -> List:
        """
        Backtrack from goal to start using parent list to construct path.

        Args:
            goal: Goal node ID

        Returns:
            Path from start to goal
        """
        path = []
        current = goal

        while current in self.parent_list:
            path.append(current)
            current = self.parent_list[current]

        path.append(current)  # Add start node
        path.reverse()
        return path

    def find_path(self, start, goal) -> Tuple[Optional[List], Optional[float]]:
        """
        Find path from start to goal using MEA* algorithm.

        Args:
            start: Start node ID
            goal: Goal node ID

        Returns:
            Tuple of (path, cost): path from start to goal and total cost,
            or (None, None) if no path exists
        """
        # Initialize
        self.open_list = set()
        self.close_list = set()
        self.parent_list = {}
        self.goal_cost = {node: float('inf') for node in self.graph.nodes()}
        self.final_cost = {node: float('inf') for node in self.graph.nodes()}

        # Calculate heuristic values for all nodes
        self.calculate_heuristic_values(goal)

        # Initialize start node
        self.open_list.add(start)
        self.goal_cost[start] = 0
        self.final_cost[start] = self.goal_cost[start] + self.heuristic_values[start]


        # Main MEA* loop
        while self.open_list:
            # Get node with minimum finalCost from openList
            current_node = min(self.open_list, key=lambda n: self.final_cost[n])

            # Check if goal reached
            if current_node == goal:
                path_found = self.backtrack_path(goal)
                total_cost = self.goal_cost[goal]
                return path_found, total_cost

            # Move current node from open to close list
            self.open_list.remove(current_node)
            self.close_list.add(current_node)

            # Process neighbors (exclude 0 and 100 unless they are the goal)
            neighbors = self.get_neighbors(current_node, goal)

            if not neighbors:
                continue

            # Calculate f_cost for each neighbor
            neighbor_costs = []
            for neighbor in neighbors:
                if neighbor in self.close_list:
                    continue

                # Calculate tentative g_cost
                tentative_g_cost = self.goal_cost[current_node] + self.get_edge_weight(current_node, neighbor)

                # Calculate f_cost
                f_cost = tentative_g_cost + self.heuristic_values[neighbor]
                neighbor_costs.append((neighbor, f_cost, tentative_g_cost))

            if neighbor_costs:
                # Find minimum neighbor
                minimum_neighbor = min(neighbor_costs, key=lambda x: x[1])
                min_neighbor_node, min_f_cost, min_g_cost = minimum_neighbor

                # Update costs for minimum neighbor
                if min_g_cost < self.goal_cost[min_neighbor_node]:
                    self.goal_cost[min_neighbor_node] = min_g_cost
                    self.final_cost[min_neighbor_node] = min_f_cost
                    self.parent_list[min_neighbor_node] = current_node
                    self.open_list.add(min_neighbor_node)

                # Add all other neighbors to close list
                for neighbor, _, _ in neighbor_costs:
                    if neighbor != min_neighbor_node:
                        self.close_list.add(neighbor)

        # No path found
        return None, None


class AStar:
    """
    A* pathfinding algorithm implementation.
    Uses Euclidean distance as the heuristic function.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initialize A* algorithm with a graph.

        Args:
            graph: NetworkX graph with node coordinates (x, y, z) and edge costs
        """
        self.graph = graph
        self.heuristic_values = {}

    def calculate_euclidean_distance(self, node1, node2) -> Optional[float]:
        """
        Calculate Euclidean distance between two nodes using their coordinates.

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Distance or None if nodes don't exist
        """
        if node1 not in self.graph.nodes() or node2 not in self.graph.nodes():
            return None

        x1, y1, z1 = self.graph.nodes[node1]['x'], self.graph.nodes[node1]['y'], self.graph.nodes[node1]['z']
        x2, y2, z2 = self.graph.nodes[node2]['x'], self.graph.nodes[node2]['y'], self.graph.nodes[node2]['z']

        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5
        return distance

    def calculate_heuristic_values(self, goal):
        """
        Pre-calculate heuristic values for all nodes to the goal.
        Uses Euclidean distance if coordinates are available.

        Args:
            goal: Goal node ID

        Raises:
            ValueError: If nodes are missing coordinate attributes (x, y, z)
        """
        self.heuristic_values = {}

        # Check if goal node has coordinates
        if 'x' not in self.graph.nodes[goal] or 'y' not in self.graph.nodes[goal] or 'z' not in self.graph.nodes[goal]:
            raise ValueError(
                f"Goal node {goal} is missing coordinate attributes (x, y, z). Cannot calculate heuristic values.")

        for node in self.graph.nodes():
            # Check if current node has coordinates
            if 'x' not in self.graph.nodes[node] or 'y' not in self.graph.nodes[node] or 'z' not in self.graph.nodes[
                node]:
                raise ValueError(
                    f"Node {node} is missing coordinate attributes (x, y, z). Cannot calculate heuristic values.")

            self.heuristic_values[node] = self.calculate_euclidean_distance(node, goal)

    def get_edge_cost(self, node1, node2) -> float:
        """
        Get the edge cost between two nodes.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Edge cost (reads 'cost' attribute from edges)
        """
        if self.graph.has_edge(node1, node2):
            edge_data = self.graph[node1][node2]
            if 'cost' in edge_data:
                return edge_data['cost']
            else:
                raise ValueError(f"Edge between {node1} and {node2} is missing 'cost' attribute")
        return float('inf')

    def get_neighbors(self, node, goal) -> List:
        """
        Get all neighbors of a node, excluding nodes 0 and 100 unless they are the goal.

        Args:
            node: Node ID
            goal: Goal node ID

        Returns:
            List of neighbor nodes (filtered)
        """
        neighbors = list(self.graph.neighbors(node))

        # Filter out nodes 0 and 100 unless they are the goal
        filtered_neighbors = []
        for neighbor in neighbors:
            if neighbor in [0, 100] and neighbor != goal:
                continue
            filtered_neighbors.append(neighbor)

        return filtered_neighbors

    def backtrack_path(self, goal, parent_dict) -> List:
        """
        Backtrack from goal to start using parent dictionary to construct path.

        Args:
            goal: Goal node ID
            parent_dict: Dictionary mapping nodes to their parents

        Returns:
            Path from start to goal
        """
        path = []
        current = goal

        while current in parent_dict:
            path.append(current)
            current = parent_dict[current]

        path.append(current)  # Add start node
        path.reverse()
        return path

    def find_path(self, start, goal) -> Tuple[Optional[List], Optional[float]]:
        """
        Find path from start to goal using A* algorithm.

        Args:
            start: Start node ID
            goal: Goal node ID

        Returns:
            Tuple of (path, cost): path from start to goal and total cost,
            or (None, None) if no path exists
        """
        # Calculate heuristic values for all nodes
        self.calculate_heuristic_values(goal)

        # Initialize data structures
        # UNVISITED set - contains all nodes except start initially
        unvisited = set(self.graph.nodes()) - {start}

        # Priority queue Q - stores (f_cost, node)
        # f_cost = g_cost + h_cost
        Q = []

        # Cost to reach each node from start (g_cost)
        cost_to_start = {node: float('inf') for node in self.graph.nodes()}
        cost_to_start[start] = 0

        # Parent dictionary for path reconstruction
        parent = {}

        # Initialize queue with start node
        f_start = cost_to_start[start] + self.heuristic_values[start]
        heapq.heappush(Q, (f_start, start))

        # Main A* loop
        while Q:
            # Pop node with minimum f_cost
            current_f, v = heapq.heappop(Q)

            # Process all neighbors of v
            neighbors = self.get_neighbors(v, goal)

            for u in neighbors:
                # Calculate tentative cost to reach u through v
                tentative_cost = cost_to_start[v] + self.get_edge_cost(v, u)

                # Update if u is unvisited OR we found a better path
                if u in unvisited or cost_to_start[u] > tentative_cost:
                    # Remove from unvisited
                    unvisited.discard(u)

                    # Update parent
                    parent[u] = v

                    # Update cost to start
                    cost_to_start[u] = tentative_cost

                    # Calculate f_cost and add/update in queue
                    f_cost = cost_to_start[u] + self.heuristic_values[u]
                    heapq.heappush(Q, (f_cost, u))

            # Check if we reached the goal
            if v == goal:
                path = self.backtrack_path(goal, parent)
                total_cost = cost_to_start[goal]
                return path, total_cost

        # No path found (FAILURE)
        return None, None


class MACO:
    """
    Modified Ant Colony Optimization (MACO) algorithm for TSP.
    Uses modified heuristic function and improved pheromone update strategy.
    """

    def __init__(self, graph: nx.Graph,
                 n_ants: int = 10,
                 n_iterations: int = 100,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 p: float = 100.0,
                 tau_0: float = 0.1):
        """
        Initialize MACO algorithm for TSP.

        Args:
            graph: NetworkX graph with edge costs and node coordinates
            n_ants: Number of ants (m)
            n_iterations: Number of iterations
            alpha: Pheromone importance parameter (α)
            beta: Heuristic importance parameter (β)
            p: Pheromone intensity coefficient
            tau_0: Initial pheromone value
        """
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.tau_0 = tau_0

        # Initialize data structures
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Pheromone matrix μ[i][j]
        self.pheromone = np.ones((self.n_nodes, self.n_nodes)) * tau_0

        # Start node for heuristic calculation
        self.start_node = None

        # Best solution tracking
        self.best_tour = None
        self.best_cost = float('inf')
        self.cost_history = []

    def get_node_coordinates(self, node) -> Tuple[float, float, float]:
        """
        Get node coordinates.

        Args:
            node: Node ID

        Returns:
            Tuple of (x, y, z) coordinates
        """
        node_data = self.graph.nodes[node]
        return node_data['x'], node_data['y'], node_data['z']

    def calculate_manhattan_distance(self, node_j, node_n) -> float:
        """
        Calculate Manhattan distance h_jn between node j and node n.
        Equation (2): h_jn(t) = |j_x - n_x| + |j_y - n_y| + |j_z - n_z|

        Args:
            node_j: Candidate node j
            node_n: Target node n (start node for TSP)

        Returns:
            Manhattan distance
        """
        x_j, y_j, z_j = self.get_node_coordinates(node_j)
        x_n, y_n, z_n = self.get_node_coordinates(node_n)

        distance = abs(x_j - x_n) + abs(y_j - y_n) + abs(z_j - z_n)
        return distance

    def calculate_improved_heuristic(self, node_i, node_j) -> float:
        """
        Calculate improved heuristic value.
        Equation (1): η'_ij(t) = 1 / (g_ij(t) + h_jn(t))

        Where:
        - g_ij(t) = direct edge cost from i to j
        - h_jn(t) = Manhattan distance from j to start node (for TSP completion)

        Args:
            node_i: Current node i
            node_j: Candidate node j

        Returns:
            Heuristic value
        """
        if node_j in (0, 100):
            return 0

        # g_ij: direct edge cost from i to j
        g_ij = self.get_edge_cost(node_i, node_j)

        # h_jn: Manhattan distance from j to 0 or 100 (guides toward completing tour)
        h_jn = min(
            self.calculate_manhattan_distance(node_j, 0),
            self.calculate_manhattan_distance(node_j, 100)
        )

        # Avoid division by zero
        denominator = g_ij + h_jn
        if denominator == 0:
            return 0

        return 1.0 / denominator

    def get_edge_cost(self, node1, node2) -> float:
        """
        Get edge cost between two nodes.

        Args:
            node1: First node
            node2: Second node

        Returns:
            Edge cost
        """
        if self.graph.has_edge(node1, node2):
            edge_data = self.graph[node1][node2]
            if 'cost' in edge_data:
                return edge_data['cost']
            else:
                raise ValueError(f"Edge between {node1} and {node2} is missing 'cost' attribute")
        return float('inf')

    def calculate_probabilities(self, current_node, unvisited_nodes: List) -> np.ndarray:
        """
        Calculate probability of moving to each unvisited node.
        P_ij^k = [μ_ij^α * η'_ij^β] / Σ[μ_il^α * η'_il^β]

        Args:
            current_node: Current node
            unvisited_nodes: List of unvisited nodes

        Returns:
            Probability array for each unvisited node
        """
        if not unvisited_nodes:
            return np.array([])

        current_idx = self.node_to_idx[current_node]

        # Calculate numerator for each unvisited node
        numerators = []
        for node in unvisited_nodes:
            node_idx = self.node_to_idx[node]

            # Pheromone component
            tau = self.pheromone[current_idx][node_idx] ** self.alpha

            # Improved heuristic component
            eta = self.calculate_improved_heuristic(current_node, node) ** self.beta

            numerators.append(tau * eta)

        numerators = np.array(numerators)

        # Calculate denominator (sum of all numerators)
        denominator = np.sum(numerators)

        # Avoid division by zero
        if denominator == 0:
            # Equal probability for all unvisited nodes
            return np.ones(len(unvisited_nodes)) / len(unvisited_nodes)

        # Calculate probabilities
        probabilities = numerators / denominator

        return probabilities

    def construct_tour_with_terminals(self, start) -> Tuple[List, float]:
        """
        Construct a TSP tour for one ant with specified start and end nodes.

        Args:
            start: Start node
            end: End node (can be same as start for closed tour)

        Returns:
            Tuple of (tour, total_cost)
            - If start == end: includes return to start in cost
            - If start != end: ends at end node (no return)
        """
        tour = [start]
        visited = {start}
        current = start
        total_cost = 0.0

        non_terminal_nodes = [n for n in self.nodes if n not in (0, 100)]

        while len(visited.intersection(non_terminal_nodes)) < len(non_terminal_nodes):
            # Get unvisited nodes
            unvisited = [
                n for n in non_terminal_nodes
                if n not in visited
            ]

            if not unvisited:
                # Should not happen, but handle it
                return None, float('inf')

            # Calculate probabilities
            probabilities = self.calculate_probabilities(current, unvisited)

            # Select next node based on probabilities
            selected_idx = np.random.choice(len(unvisited), p=probabilities)
            next_node = unvisited[selected_idx]

            # Update tour and cost
            edge_cost = self.get_edge_cost(current, next_node)
            if edge_cost == float('inf'):
                # No edge exists
                return None, float('inf')

            tour.append(next_node)
            visited.add(next_node)
            total_cost += edge_cost
            current = next_node

        # Force termination at a terminal node
        possible_ends = [0, 100]

        if current in (0, 100) and current != start:
            return None, float('inf')

        end_node = np.random.choice(possible_ends)

        edge_cost = self.get_edge_cost(current, end_node)
        if edge_cost == float('inf'):
            return None, float('inf')

        tour.append(end_node)
        total_cost += edge_cost

        return tour, total_cost

    def count_ants_on_tour(self, target_tour: List, all_tours: List[Tuple[List, float]]) -> int:
        """
        Count how many ants traveled on the same tour.

        Args:
            target_tour: The tour to check
            all_tours: All tours from all ants

        Returns:
            Number of ants that traveled this exact tour
        """
        if target_tour is None:
            return 0

        count = 0
        for tour, cost in all_tours:
            if tour is not None and tour == target_tour:
                count += 1

        return count

    def update_pheromones_improved(self, all_tours: List[Tuple[List, float]]):
        """
        Update pheromone levels using improved global strategy.

        At the end of each iteration:
        1. Find THE best tour (lowest cost) among all ants
        2. Find THE worst tour (highest cost) among all ants
        3. Count number of ants that traveled on best tour (N_g)
        4. Count number of ants that traveled on worst tour (N_b)
        5. Apply reward to best tour and penalty to worst tour

        Equations (3-5):
        μ_ij^new(t+1) = μ_ij(t) + N_g × Δμ[g_ij] - N_b × Δμ[b_ij]

        Where:
        - N_g = number of ants that traveled the best tour
        - N_b = number of ants that traveled the worst tour
        - Δμ[g_ij] = p/l_g if (i,j) in THE best tour, else 0
        - Δμ[b_ij] = p/l_b if (i,j) in THE worst tour, else 0

        Args:
            all_tours: List of (tour, cost) tuples from all ants
            start: Start node
            end: End node
        """
        # Filter out invalid tours
        valid_tours = [(tour, cost) for tour, cost in all_tours
                       if tour is not None and cost != float('inf')]

        if not valid_tours:
            return

        # Sort tours by cost (ascending)
        sorted_tours = sorted(valid_tours, key=lambda x: x[1])

        # Get THE best tour (lowest cost)
        best_tour, l_g = sorted_tours[0]

        # Get THE worst tour (highest cost)
        worst_tour, l_b = sorted_tours[-1]

        # Count number of ants on best and worst tours
        N_g_dynamic = self.count_ants_on_tour(best_tour, all_tours)
        N_b_dynamic = self.count_ants_on_tour(worst_tour, all_tours)

        # Reward THE best tour (Equation 4)
        if N_g_dynamic > 0:
            delta_g = self.p / l_g

            # Update edges in tour
            tour_edges = list(zip(best_tour, best_tour[1:]))

            for node_i, node_j in tour_edges:
                idx_i = self.node_to_idx[node_i]
                idx_j = self.node_to_idx[node_j]

                # Add reward
                self.pheromone[idx_i][idx_j] += N_g_dynamic * delta_g

        # Penalize THE worst tour (Equation 5)
        if N_b_dynamic > 0:
            delta_b = self.p / l_b

            # Update edges in tour
            tour_edges = list(zip(worst_tour, worst_tour[1:]))

            for node_i, node_j in tour_edges:
                idx_i = self.node_to_idx[node_i]
                idx_j = self.node_to_idx[node_j]

                # Subtract penalty
                self.pheromone[idx_i][idx_j] -= N_b_dynamic * delta_b

        # Ensure pheromone levels don't go below minimum threshold
        self.pheromone = np.maximum(self.pheromone, self.tau_0)

    def solve_tsp(self) -> Tuple[Optional[List], Optional[float]]:
        """
        Solve TSP starting from a given node and ending at another node.

        Args:
            start: Start node for the tour
            end: End node for the tour (can be same as start for closed tour)

        Returns:
            Tuple of (best_tour, best_cost)
            - If start == end: tour returns to start (closed tour)
            - If start != end: tour ends at end node (open tour)
        """

        for iteration in range(self.n_iterations):
            # All ants construct tours
            all_tours = []

            for ant in range(self.n_ants):
                start_node = np.random.choice([0, 100])
                tour, cost = self.construct_tour_with_terminals(start_node)
                all_tours.append((tour, cost))

                # Update best solution
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_tour = tour

            # Update pheromones using improved strategy
            self.update_pheromones_improved(all_tours)

            # Track progress
            self.cost_history.append(self.best_cost)

        return self.best_tour, self.best_cost