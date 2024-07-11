import numpy as np
import heapq

class Planner:
    '''
    A graph-based planner
    Implementation of the Dijkstra algorithm to obtain the path among existing rooms
    '''

    def __init__(self,
                 areas,
                 intersections,
                 viapoints):
        self.areas = areas
        self.intersections = intersections
        self.viapoints = viapoints
        self.graph = self.create_graph(areas, intersections, viapoints)
        self.path = []
        self.path_index = 0

    def create_graph(self, areas, intersections, viapoints):
        graph = {area: {} for area in areas}
        for (area1, area2) in intersections:
            if (area1, area2) in viapoints:
                graph[area1][area2] = viapoints[(area1, area2)]
                graph[area2][area1] = viapoints[(area1, area2)]
            elif (area2, area1) in viapoints:
                graph[area1][area2] = viapoints[(area2, area1)]
                graph[area2][area1] = viapoints[(area2, area1)]
        return graph
    
    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def dijkstra(self, start, goal):
        queue = [(0, start, [])] # (cost, current_node, path)
        seen = set()
        min_dist = {start: 0}

        while queue:
            (cost, node, path) = heapq.heappop(queue)

            if node in seen:
                continue

            path = path + [node]
            seen.add(node)

            if node == goal:
                return (cost, path)

            for neighbor, viapoint in self.graph[node].items():
                if neighbor in seen:
                    continue
                prev_cost = min_dist.get(neighbor, float('inf'))
                new_cost = cost + 1  # Base cost for moving to a new node
                
                if len(path) > 1:
                    last_viapoint = (self.viapoints.get((path[-2], node)) 
                                     if (path[-2], node) in self.viapoints 
                                     else self.viapoints.get((node, path[-2])))
                    new_cost += self.euclidean_distance(last_viapoint, viapoint)

                if new_cost < prev_cost:
                    min_dist[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path))

        return float("inf"), []
    
    def compute_path(self, start, goal):
        cost, path = self.dijkstra(start, goal)
        self.path = path
        self.path_index = 0
        return cost, path
    
    def get_next_step(self):
        if self.path_index >= len(self.path) - 1:
            return self.path[-1], None
        
        current_node = self.path[self.path_index]
        next_node = self.path[self.path_index + 1]
        viapoint = (self.viapoints.get((current_node, next_node)) 
                    if (current_node, next_node) in self.viapoints 
                    else self.viapoints.get((next_node, current_node)))

        self.path_index += 1
        return next_node, viapoint
