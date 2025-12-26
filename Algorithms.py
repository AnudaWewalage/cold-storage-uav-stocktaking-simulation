import networkx as nx
from typing import List, Optional, Dict, Set, Tuple
import math
import pickle

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