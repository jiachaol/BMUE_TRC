#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:45:12 2021

@author: liujiachao
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import networkx as nx
import pickle
import scipy as sp
import queue

#sys.path
os.chdir('/Users/liujiachao/BMUE/src')
from network_handler import *
from graph_modifier import *
from path_generator_new import *
from solver_v2 import *
# from solver import *
from fn_and_const import *
import warnings
warnings.filterwarnings("ignore")
import time

# network handle
tester_net = network_handler('6link', '/Users/liujiachao/BMUE')
tester_net.build_network_graph()
print("--- network graph ---")
print (nx.info(tester_net.network_graph))

tester_net.build_demand_graph()
print("--- demand graph ---")
print (nx.info(tester_net.demand_graph))

tester_modified = graph_modifier()
tester_modified.build_modified_network(tester_net,1)
print("--- modified net ---")
print(nx.info(tester_modified.modified_graph))

tester_path = path_generator(tester_modified, 1.5, 5)

self = tester_path

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
    (u,v) = list(G_demand.edges)[0]
    #print(time.asctime(time.localtime(time.time())))
    a = time.time()
    print("OD pair: " + str(u) + "-" + str(v) + " begin")
    print(time.asctime(time.localtime(a)))
    # drive_path, len_drive_path, ridehailing_path, len_ridehailing_path = self._generate_OD_paths(u,v,shortest_paths_length[u])
    
    O,D,lengths = u,v,shortest_paths_length[u]
    
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
        #print(n,l,path)
        if n == O and path[0] == D:
            if self.max_paths is None or len(drive_path) < self.max_paths:
                drive_path.append(path.copy())
                len_drive_path.append(l)
        
        if n == O and path[0] in G_demand.nodes[D]['neighborhood'].keys():
            if self.max_paths is None or len(drive_path) < self.max_paths:
                drive_path.append(path.copy())
                len_drive_path.append(l)
        
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
self.paths = all_drive_path + all_ridehailing_path
self.vec_len_paths = sp.sparse.csr_matrix(np.array(len_all_drive_path + len_all_ridehailing_path)).transpose()
