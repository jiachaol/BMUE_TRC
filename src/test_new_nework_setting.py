#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:58:11 2021

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
tester_modified.build_modified_network(tester_net, 1)
network = tester_net
print("--- modified net ---")
print(nx.info(tester_modified.modified_graph))

for node in list(tester_modified.network.demand_graph.nodes):
    print(str(node) + " neighborhood num = " + str(len(tester_modified.network.demand_graph.nodes[node]['neighborhood'])))

tester_path = path_generator(tester_modified, 10, 10)

refconsts = fn_and_const()
refconsts.const_dest_parking = 20.0
refconsts.const_curb_cap = 50.0

S = VI_solver_v2(tester_path,refconsts)

for (u,v) in list(S.demand_graph.edges):
    S.demand_graph.edges[(u, v)]['volume'] = 6000

S.solve_sue(5e-5, 2000, 0.5, True, True)

print(S.vec_flow.transpose()[0])
print(S.vec_cost.transpose()[0])

# print(S.vec_flow)

# S.paths

# S.vec_cost

# self = S
# link_path_matrix = S.mat_link_in_path
# link_flow = link_path_matrix * vec_flow


# link_flow.to_csv("/Users/liujiachao/BMUE/link_flow.csv")
# np.save("/Users/liujiachao/BMUE/link_flow.npy",link_flow)


S.solve_curbcharge(step_size = 1, max_iterations = 50, sue_precision = 1e-4, sue_max_iterations = 2000, \
                         sue_step_size = 0.5, print_gap = True, nneg = True, alg_type = 'adag')
    

figsize(10, 6)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(range(1,len(S.charge_gap_list)+1), S.charge_gap_list, color='orange', linewidth=2)
plt.xlabel("iteration",fontsize=16, family = 'arial')
plt.ylabel("total social cost", fontsize=16, family = 'arial')
plt.xlim(1,len(S.charge_gap_list)+1)
plt.show()


flow_init = np.array([[0] for x in range(S.num_path)], dtype=float)
for (u,v) in list(S.demand_graph.edges):
    # (u,v) = list(self.demand_graph.edges)[0]
    a,b,c,d = S.idx_path_between_OD[u][v]
    init_flow = S.demand_graph.edges[(u, v)]['volume']/(b - a + d - c)
    for i in range(a,b):
        flow_init[i] = init_flow
        # print(i)
    for j in range(c,d):
        flow_init[j] = init_flow
        
S.solve_so(flow_init, max_iterations = 1000, plot_gap = False)

figsize(10, 6)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'out'
plt.plot(range(1,len(S.so_sc_list)+1), S.so_sc_list, color='orange', linewidth=2)
plt.xlabel("iteration",fontsize=16, family = 'arial')
plt.ylabel("total social cost", fontsize=16, family = 'arial')
plt.xlim(0,len(S.so_sc_list)+1)
plt.show()

print("result: demand = " + str(S.demand_graph.edges[(1, 6)]['volume']))
print("parking cost = " + str(S.const_dest_parking))
print("curb cap = " + str(S.const_curb_cap))
print("Optimal pricing = " + str(S.vec_charge[S.charge_gap_list.index(min(S.charge_gap_list))][0,0]))
print("Optimal charge TSC = " + str(min(S.charge_gap_list)))

# print("Flow = " + str(S.flow_list[S.charge_gap_list.index(min(S.charge_gap_list))]))
print("System optimum TSC = " + str(min(S.so_sc_list)))
print("BMUE TSC = " + str(S.charge_gap_list[0]))
# print(S.vec_flow.transpose())

sue_path_flow = S.sue_flow_list

path_k1 = []
path_k2 = []
path_k5 = []
path_k6 = []

for i in range(0,len(sue_path_flow)):
    path_k1.append(sue_path_flow[i][0,0])
    path_k2.append(sue_path_flow[i][0,1])
    path_k5.append(sue_path_flow[i][0,4])
    path_k6.append(sue_path_flow[i][0,5])

figsize(9, 5)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.plot(range(0,len(path_k1)), path_k1, color='red', linewidth=1, label = "flow on path k3")
plt.plot(range(0,len(path_k1)), path_k2, color='blue', linewidth=1, label = "flow on path k4")
plt.plot(range(0,len(path_k1)), path_k5, color='green', linewidth=1, label = "flow on path k5")
plt.plot(range(0,len(path_k1)), path_k6, color='orange', linewidth=1, label = "flow on path k6")
plt.xlabel("Iterations",fontsize=13, family = 'arial')
plt.ylabel("Path Flow", fontsize=13, family = 'arial')
# plt.title("Evolution of flows on 4 paths in 6-link network", fontsize=14, family = 'arial')
plt.xlim(0,300)
plt.legend()
plt.show()

# print(S.vec_flow.transpose())
# print(S.vec_cost.transpose())

# Delta = S.mat_link_in_path.todense()

# Lambda = S.mat_path_between_OD.todense()

# mat_raw = np.concatenate((Delta, Lambda), axis = 0)

# path_flow = S.vec_flow

# link_flow = Delta * path_flow

