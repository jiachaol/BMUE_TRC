#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:43:32 2021

@author: liujiachao

This is the script considering destination choice for driving mode


"""

import numpy as np
import scipy as sp
import queue
import networkx as nx
import time

class path_generator():
    def __init__(self,network,path_len_cap_ratio,max_paths = None):
        self.network = network
        self.len_cap = path_len_cap_ratio
        self.paths = []
        self.paths_raw = []
        self.max_paths = max_paths
        self.mat_link_in_path = None
        self.mat_path_start_from_link = None
        self.mat_path_end_at_link = None
        self.idx_path_between_OD = dict()
        self.mat_path_between_OD = None
        self.mat_Dpath_between_OD = None
        self.mat_Rpath_between_OD = None
        self.num_drive_paths = 0
        self.num_ridehailing_paths = 0
        self.vec_OD = None
        self.vec_len_paths = []
        self.vec_walking_dist = None
        self.idx_to_OD = dict()
        for node in list(self.network.network.demand_graph.nodes):
            if self.network.network.demand_graph.out_degree(node) > 0:
                self.idx_path_between_OD[node] = dict()
        self._generate_paths()
        self._generate_vectors()

    def __str__(self):
        ret_str = ""
        for (u,v) in self.network.network.network_graph.edges:
            e = self.network.network.network_graph.edges[u,v]
            ret_str += "Edge No. " + str(e["index"]) + ": (" + str(u) + "," + str(v) + ")\n"
        for i in range(len(self.paths)):
            ret_str += "Path No. " + str(i) + ": " + str(self.paths[i]) + "\n"
        ret_str += "link in path \n" + str(self.mat_link_in_path.todense()) + "\n"
        ret_str += "path start from link \n" + str(self.mat_path_start_from_link.todense()) + "\n"
        ret_str += "path end at link \n" + str(self.mat_path_end_at_link.todense()) + "\n"
        ret_str += "path between OD \n" + str(self.mat_path_between_OD.todense()) + "\n"
        ret_str += "Dpath between OD \n" + str(self.mat_Dpath_between_OD.todense()) + "\n"
        ret_str += "Rpath between OD \n" + str(self.mat_Rpath_between_OD.todense()) + "\n"
        return ret_str

    def _generate_OD_paths(self,O,D,lengths):
        G_demand = self.network.network.demand_graph
        G_modified = self.network.modified_graph
        cap_length = lengths[D] * self.len_cap
        drive_path = []
        len_drive_path = []
        ridehailing_path = []
        len_ridehailing_path = []
        Q = queue.Queue()
        Q.put((D,0,[D],{D}))
        for node in G_demand.nodes[D]['neighborhood'].keys():
            if node != D:
                Q.put((node,0,[node],{node}))
        while not Q.empty():
            if not (self.max_paths is None) and len(drive_path) >= self.max_paths and len(ridehailing_path) >= self.max_paths:
                break
            (n,l,path,visited) = Q.get()
            
            
            # from O to D
            if n == O and path[0] == D:
                if self.max_paths is None or len(drive_path) < self.max_paths:
                    drive_path.append(path.copy())
                    len_drive_path.append(l)
            
            # from O to curb node (destination choice)
            if n == O and path[0] in G_demand.nodes[D]['neighborhood'].keys():
                if self.max_paths is None or len(drive_path) < self.max_paths:
                    drive_path.append(path.copy())
                    len_drive_path.append(l)
            
            # ridehailing path from curb to curb        
            if not (G_demand.nodes[O]['neighborhood'].get(n) is None or G_demand.nodes[D]['neighborhood'].get(path[0]) is None):
                if self.max_paths is None or len(ridehailing_path) < self.max_paths:
                    ridehailing_path.append(path.copy())
                    len_ridehailing_path.append(l)
                    
            if n == 0:
                continue
            
            for w in [x[0] for x in list(G_modified.in_edges(n))]:
                if w in visited:
                    continue
                e = G_modified.edges[(w,n)]
                leng = l + e['length']
                if w in lengths: # added by Jiachao
                    if leng + lengths[w] < cap_length:
                        new_path = path.copy()
                        new_path.append(w)
                        new_visited = visited.copy()
                        new_visited.add(w)
                        Q.put((w, leng, new_path, new_visited))
                    
        return drive_path, len_drive_path, ridehailing_path, len_ridehailing_path

    def _generate_paths(self):
        shortest_paths_length = dict()
        G_demand = self.network.network.demand_graph
        G_modified = self.network.modified_graph
        all_drive_path = []
        all_ridehailing_path = []
        len_all_drive_path = []
        len_all_ridehailing_path = []
        for node in list(G_demand.nodes):
            if G_demand.out_degree(node) > 0:
                shortest_paths_length[node] = \
                    nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(G_modified,node,weight = 'length')
        for (u,v) in list(G_demand.edges):
            #print(time.asctime(time.localtime(time.time())))
            a = time.time()
            print("OD pair: " + str(u) + "-" + str(v) + " begin")
            # print(time.asctime(time.localtime(a)))
            drive_path, len_drive_path, ridehailing_path, len_ridehailing_path = self._generate_OD_paths(u,v,shortest_paths_length[u])
            
            idx_d = len(all_drive_path)
            idx_r = len(all_ridehailing_path)
            l_drive = len(drive_path)
            l_ridehailing = len(ridehailing_path)

            self.idx_path_between_OD[u][v] = (idx_d,idx_d+l_drive,idx_r,idx_r+l_ridehailing)

            G_demand.edges[(u,v)]['num_paths'] = l_drive + l_ridehailing
            all_drive_path = all_drive_path + drive_path
            len_all_drive_path = len_all_drive_path + len_drive_path
            all_ridehailing_path = all_ridehailing_path + ridehailing_path
            len_all_ridehailing_path = len_all_ridehailing_path + len_ridehailing_path
            b = time.time()
            print("OD pair: " + str(u) + "-" + str(v) + "  Running time: " + str(round(b - a,2)) + "s")
            
            
        self.num_drive_paths = len(all_drive_path)
        self.num_ridehailing_paths = len(all_ridehailing_path)
        self.paths_raw = all_drive_path + all_ridehailing_path
        self.vec_len_paths = sp.sparse.csr_matrix(np.array(len_all_drive_path + len_all_ridehailing_path)).transpose()

    def _generate_vectors(self):
        G_network = self.network.network.network_graph
        G_modified = self.network.modified_graph
        G_demand = self.network.network.demand_graph
        num_link = len(G_network.edges)
        num_path = len(self.paths_raw)
        num_demand = len(G_demand.edges)
        link_in_path = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        path_start_from_link = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        path_end_at_link = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        Dpath_between_OD = sp.sparse.dok_matrix((num_demand,num_path), dtype = np.float32)
        Rpath_between_OD = sp.sparse.dok_matrix((num_demand, num_path), dtype=np.float32)
        ODvector = sp.sparse.dok_matrix((num_demand,1), dtype = np.float32)
        self.vec_walking_dist = sp.sparse.dok_matrix((num_path,1), dtype = np.float32)
        
        # reverse path and delete the curb node in the middle of paths
        for i in range(num_path):
            # i = 0
            path = self.paths_raw[i]
            path.reverse()
            node_remove = []
            
            if not G_modified.nodes[path[0]].get('isOriginal', True):
                link_in_path[G_modified.nodes[path[0]]['edgeidx'],i] = 1
                             
                if (i > self.num_drive_paths)|(i == self.num_drive_paths):
                    path_start_from_link[G_modified.nodes[path[0]]['edgeidx'],i] = 1
                
                
            if not G_modified.nodes[path[len(path)-1]].get('isOriginal', True):
                link_in_path[G_modified.nodes[path[len(path)-1]]['edgeidx'], i] = 1
                if (i > self.num_drive_paths)|(i == self.num_drive_paths):
                    path_end_at_link[G_modified.nodes[path[len(path)-1]]['edgeidx'], i] = 1
                
            for j in range(len(path)-2,0,-1):
                v = path[j]
                if not G_modified.nodes[v].get('isOriginal', True):
                    link_in_path[G_modified.nodes[v]['edgeidx'], i] = 1
                    node_remove.append(j)
            for j in node_remove:
                del path[j]
            
            self.paths.append(path)
        
        idx_OD = 0
        for O in self.idx_path_between_OD.keys():
            for D in self.idx_path_between_OD[O].keys():
                (idx_db,idx_de,idx_rb,idx_re) = self.idx_path_between_OD[O][D]
                for i in range(idx_db,idx_de):
                    Dpath_between_OD[idx_OD,i] = 1
                    path = self.paths[i]
                    
                    dis = 0
                    if path[0] != O:
                        dis += G_demand.nodes[O]['neighborhood'][path[0]]
                    if path[-1] != D:
                        dis += G_demand.nodes[D]['neighborhood'][path[-1]]
                    if (path[0] != O)|(path[-1] != D):
                        self.vec_walking_dist[i,0] = dis
                        
                for i in range(self.num_drive_paths + idx_rb, self.num_drive_paths + idx_re):
                    Rpath_between_OD[idx_OD,i] = 1
                    path = self.paths[i]
                    self.vec_walking_dist[i,0] = G_demand.nodes[O]['neighborhood'][path[0]] + G_demand.nodes[D]['neighborhood'][path[-1]]
                    
                self.idx_path_between_OD[O][D] = (idx_db,idx_de,self.num_drive_paths+idx_rb,self.num_drive_paths+idx_re)
                ODvector[idx_OD,0] = G_demand.edges[(O,D)]['volume']
                self.idx_to_OD[idx_OD] = (O,D)
                G_demand.edges[(O,D)]['idx'] = idx_OD
                idx_OD += 1
        
        self.mat_Dpath_between_OD = Dpath_between_OD.tocsr()
        self.mat_Rpath_between_OD = Rpath_between_OD.tocsr()
        self.mat_path_between_OD = Dpath_between_OD + Rpath_between_OD
        self.mat_link_in_path = link_in_path.tocsr()
        self.mat_path_end_at_link = path_end_at_link.tocsr()
        self.mat_path_start_from_link = path_start_from_link.tocsr()
        self.vec_OD = ODvector.tocsr()
