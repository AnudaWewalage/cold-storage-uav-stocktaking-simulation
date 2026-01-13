MAinimport random
import networkx as nx
from typing import List
from Algorithms import AStar


def generate_unique_random_nodes_explicit(n):
    """
    Explicit implementation with step-by-step verification.
    """
    # Define ranges
    range1 = list(range(1, 33))  # 1-32
    range2 = list(range(101, 133))  # 101-132
    all_nodes = range1 + range2

    if n > len(all_nodes):
        raise ValueError(f"Cannot generate {n} unique nodes. Maximum is {len(all_nodes)}.")

    if n < 0:
        raise ValueError(f"Number of nodes must be positive, got {n}.")

    # Shuffle all nodes and take first n
    random.shuffle(all_nodes)
    selected_nodes = all_nodes[:n]

    # Double-check uniqueness
    if len(set(selected_nodes)) != len(selected_nodes):
        # This should never happen with this implementation
        raise RuntimeError("Duplicate nodes detected - this is unexpected!")

    return selected_nodes


def build_reduced_digraph(
    path_finder: any,
    node_list: List[int]
) -> nx.DiGraph:

    reduced_nodes = set(node_list)
    reduced_nodes.update([0, 100])

    G_full = path_finder.graph   # or whatever graph A* is using
    G_reduced = nx.DiGraph()

    # Copy nodes WITH attributes
    for n in reduced_nodes:
        if n in G_full.nodes:
            G_reduced.add_node(n, **G_full.nodes[n])
        else:
            G_reduced.add_node(n)

    # Directed: run A* for every ordered pair
    for u in reduced_nodes:
        for v in reduced_nodes:
            if u == v:
                continue

            path, cost = path_finder.find_path(u, v)
            if path is not None and cost is not None:
                G_reduced.add_edge(u, v, cost=cost)

    return G_reduced

def solve_full_path_with_intermediates(G, must_visit_nodes, weight='weight'):
    """
    Solves the multi-stop shortest path and returns every intermediate node.
    Returns to the start node at the end.
    """
    if not must_visit_nodes:
        return [], 0

    full_path = []
    total_cost = 0

    # 1. Define the sequence of stops (A -> B -> C -> ... -> A)
    # We append the first node to the end to ensure it loops back
    stops = list(must_visit_nodes)
    stops.append(must_visit_nodes[0])

    try:
        for i in range(len(stops) - 1):
            start_stop = stops[i]
            end_stop = stops[i+1]

            # 2. Calculate shortest path for this specific segment
            # This uses Dijkstra's theory under the hood
            segment_path = nx.shortest_path(G, source=start_stop, target=end_stop, weight=weight)
            segment_cost = nx.path_weight(G, segment_path, weight=weight)

            # 3. Stitch the paths
            if i == 0:
                # For the very first segment, take the whole path
                full_path.extend(segment_path)
            else:
                # For subsequent segments, skip the first node (it's the same as the last of previous)
                full_path.extend(segment_path[1:])

            total_cost += segment_cost

        return full_path, total_cost

    except nx.NetworkXNoPath:
        print(f"Error: No path exists between {start_stop} and {end_stop}")
        return [], -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], -1

def find_optimal_path_for_mixed_racks(nodes, G_rack1, G_rack2):
    if not nodes:
        return None

    rack1_nodes = [0]
    rack2_nodes = [100]

    for node in nodes:

        if node <= 32:
            rack1_nodes.append(node)
        elif node > 32:
            rack2_nodes.append(node)

    path_rack1, cost_rack1 = solve_full_path_with_intermediates(G_rack1, rack1_nodes,'cost')
    path_rack2, cost_rack2 = solve_full_path_with_intermediates(G_rack2, rack2_nodes, 'cost')

    total_cost = cost_rack1 + cost_rack2

    return path_rack1, path_rack2, total_cost
