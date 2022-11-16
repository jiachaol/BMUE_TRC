import queue
import networkx as nx

class graph_modifier():
    def __init__(self):
        self.network = None
        self.modified_graph = None
        self.r_neighborhood = 0
        self.max_term_deg = 0
        self.max_node = 0

    def build_modified_network(self, network, distance_threshold = 0.1, max_terminal_degree = 2):
        self.network = network
        self.max_node = network.max_node
        self.modified_graph = None
        self.r_neighborhood = distance_threshold
        self.max_term_deg = max_terminal_degree
        self._add_link_node()
        self._compute_neighborhood()

    def _add_link_node(self):
        G = nx.DiGraph()
        for (u, v, w) in self.network.network_graph.edges.data():
            idx = w['index']
            len_u = w['length'] * w['curb']
            len_v = w['length'] * (1 - w['curb'])
            G.add_node(self.max_node, fromNode = u, toNode = v, edgeidx = idx, isOriginal = False)
            G.add_edge(u, self.max_node, length = len_u, index = idx)
            G.add_edge(self.max_node, v, length = len_v, index = idx)
            self.max_node += 1
        self.modified_graph = G

    def _find_neighborhood(self, node):
        Q = queue.Queue()
        neighbors = dict()
        G = self.modified_graph
        neighbors[node] = 0
        Q.put((node,0))
        while not Q.empty():
            (n, l) = Q.get()
            frontier = [x[1] for x in list(G.out_edges(n))]
            for v in frontier:
                e = G.edges[(n,v)]
                newlen = l + e['length']
                if newlen < self.r_neighborhood:
                    if (neighbors.get(v) is None) or (newlen < neighbors[v]):
                        neighbors[v] = newlen
                        Q.put((v, newlen))
            frontier = [x[0] for x in list(G.in_edges(n))]
            for v in frontier:
                e = G.edges[(v,n)]
                newlen = l + e['length']
                if newlen < self.r_neighborhood:
                    if (neighbors.get(v) is None) or (newlen < neighbors[v]):
                        neighbors[v] = newlen
                        Q.put((v, newlen))
        neighborhood = dict()
        for v in neighbors.keys():
            if G.in_degree(v) + G.out_degree(v) <= self.max_term_deg and not G.nodes[v].get('isOriginal', True):
                neighborhood[v] = neighbors[v]
        return neighborhood

    def _compute_neighborhood(self):
        for node in list(self.network.demand_graph.nodes):
            neighbors = self._find_neighborhood(node)
            self.network.demand_graph.nodes[node]['neighborhood'] = neighbors
