import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import time
import networkx as nx
import matlab
import matlab.engine

class VI_solver_v2():
    # All vectors are Col vectors unless otherwise specified
    def __init__(self,network,fn_and_const):
        self.network_graph = network.network.network.network_graph
        self.demand_graph = network.network.network.demand_graph
        self.modified_graph = network.network.modified_graph
        self.num_link = len(self.network_graph.edges)
        self.num_demand = len(self.demand_graph.edges)
        self.num_path = len(network.paths)
        self.num_drive_path = network.num_drive_paths
        self.num_ridehailing_path = network.num_ridehailing_paths
        self.idx_path_between_OD = network.idx_path_between_OD
        self.paths = network.paths
        self.paths_raw = network.paths_raw
        # Sparse Matrices
        self.mat_link_in_path = network.mat_link_in_path
        self.mat_path_start_from_link = network.mat_path_start_from_link
        self.mat_path_end_at_link = network.mat_path_end_at_link
        self.mat_path_between_OD = network.mat_path_between_OD
        self.mat_Dpath_between_OD = network.mat_Dpath_between_OD
        self.mat_Rpath_between_OD = network.mat_Rpath_between_OD
        # Load Constants
        self.const_vtime = fn_and_const.const_vtime
        self.const_driving_cost = fn_and_const.const_driving_cost
        self.const_dest_parking = fn_and_const.const_dest_parking
        self.const_rhfare_time = fn_and_const.const_rhfare_time
        self.const_rhfare_dist = fn_and_const.const_rhfare_dist
        self.const_rhfare_overhead = fn_and_const.const_rhfare_overhead
        self.const_walking_speed = fn_and_const.const_walking_speed
        self.const_curb_cap = fn_and_const.const_curb_cap
        self.const_t_commute = fn_and_const.const_t_commute
        self.const_t_curbstop = fn_and_const.const_t_curbstop
        self.const_alpha_D = fn_and_const.const_alpha_D
        self.const_alpha_R = fn_and_const.const_alpha_R
        self.const_beta = fn_and_const.const_beta
        self.const_initial_curb_charge = fn_and_const.const_initial_curb_charge
        self.q_model_rate_threshold = fn_and_const.q_model_rate_threshold
        self.charge_limit = fn_and_const.charge_limit
        self.activation_coefficient = fn_and_const.activation_coefficient
        
        # Dense Vectors
        self.vec_OD = np.array(network.vec_OD.todense())
        self.vec_len_paths = np.array(network.vec_len_paths.todense())
        self.vec_walking_dist = np.array(network.vec_walking_dist.todense())
        self.vec_const_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_time_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_flow = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_change_flow = np.array([[0] for x in range(self.num_path)], dtype=float) # ljc
        self.vec_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_social_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_lambda_1 = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.vec_lambda_2 = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.vec_freeflow = np.array([[0] for x in range(self.num_link)], dtype=float) # float ljc
        self.vec_inv_link_cap = np.array([[0] for x in range(self.num_link)], dtype=float) # float ljc
        self.vec_link_len = np.array([[0] for x in range(self.num_link)], dtype=float) # float ljc
        self.vec_curb_cap = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.vec_curb_charge = np.array([[self.const_initial_curb_charge] for x in range(self.num_link)], dtype=float)
        self.vec_queue_length = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.grad = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.grad_sum = np.array([[0] for x in range(self.num_link)], dtype=float)
        self.sue_gap_list = list()
        self.sue_flow_list = list()
        self.sue_cost_list = list()
        self.charge_gap_list = list()
        self.flow_list = list()
        self.cost_list = list()
        self.vec_charge = list()
        self.auxiliary_flow = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.so_gap_list = list()
        self.so_sc_list = list()
        
        
        for (u,v) in list(self.network_graph.edges):
            #(u,v) = list(self.network_graph.edges)[0]
            e = self.network_graph.edges[(u,v)]
            # link freeflow const
            self.vec_freeflow[e['index'],0] = e['fft']
            # 1/capacity const
            self.vec_inv_link_cap[e['index'],0] = 1.0 / e['cap'] 
            # link length const
            self.vec_link_len[e['index'],0] = e['length']
        
        for i in range(self.num_link):
            self.vec_curb_cap[i,0] = self.const_curb_cap
            
        vec_const = np.array([[0] for x in range(self.num_path)], dtype=float)
        vec_distance_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        for i in range(self.num_drive_path):
            vec_distance_cost[i, 0] = self.const_driving_cost # The per mile unit cost constant for driving = 1.5
            vec_const[i, 0] = self.const_dest_parking # destination parking price = 20
            self.vec_time_cost[i, 0] = self.const_vtime # Time cost constant
        for i in range(self.num_drive_path, self.num_path):
            vec_distance_cost[i, 0] = self.const_rhfare_dist # 1.75
            vec_const[i, 0] = self.const_rhfare_overhead # 2.55
            self.vec_time_cost[i, 0] = self.const_vtime + self.const_rhfare_time
        vec_walk_time = self.vec_walking_dist * (1.0 / self.const_walking_speed)
        self.vec_const_cost = vec_const + np.multiply(vec_distance_cost, self.vec_len_paths) + vec_walk_time * self.const_vtime
        
        

    def __str__(self):
        return ""
        
    def _find_min_range(self,vec,l,r):
        min_idx = l
        for i in range(l,r):
            if vec[i,0] < vec[min_idx,0]:
                min_idx = i
        return min_idx
    
    def _find_min_so(self,vec,l,r):
        min_idx = l
        for i in range(l,r):
            if vec[i] < vec[min_idx]:
                min_idx = i
        return min_idx

    def _compute_logit_adjustment(self):
        vec_adjustment = np.array([0 for x in range(self.num_path)])
        for (u, v) in list(self.demand_graph.edges):
            e = self.demand_graph.edges[(u, v)]
            vec_ODpaths_D = self.mat_Dpath_between_OD.getrow(e['idx'])
            vec_ODpaths_R = self.mat_Rpath_between_OD.getrow(e['idx'])
            q_D = vec_ODpaths_D.dot(self.vec_flow)[0, 0]
            q_R = vec_ODpaths_R.dot(self.vec_flow)[0, 0]
            logit_term_D = (self.const_alpha_D + np.log(q_D)) / self.const_beta
            logit_term_R = (self.const_alpha_R + np.log(q_R)) / self.const_beta
            vec_adjustment = vec_adjustment + vec_ODpaths_D.multiply(logit_term_D) + vec_ODpaths_R.multiply(logit_term_R)
        return np.transpose(vec_adjustment)

    def compute_sue_gap(self):
        if self.vec_cost is None:
            return None
        vec_min_cost = np.array([[0] for x in range(self.num_path)], dtype = float)
        for (u,v) in list(self.demand_graph.edges):
            e = self.demand_graph.edges[(u,v)]
            vec_ODpaths_D = self.mat_Dpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths_R = self.mat_Rpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths = vec_ODpaths_D + vec_ODpaths_R
            vec_temp_D = np.array(vec_ODpaths_D.multiply(self.vec_cost).todense())
            vec_temp_R = np.array(vec_ODpaths_R.multiply(self.vec_cost).todense())
            min_D = self._find_min_range(vec_temp_D, self.idx_path_between_OD[u][v][0], self.idx_path_between_OD[u][v][1])
            if self.idx_path_between_OD[u][v][2] != self.idx_path_between_OD[u][v][3]:
                min_R = self._find_min_range(vec_temp_R, self.idx_path_between_OD[u][v][2], self.idx_path_between_OD[u][v][3])
                min = vec_temp_D[min_D, 0]
                if vec_temp_R[min_R, 0] < min:
                    min = vec_temp_R[min_R, 0]
            else:
                min = vec_temp_D[min_D, 0]
            vec_min_cost += vec_ODpaths.multiply(min)
        return (np.transpose(self.vec_cost - vec_min_cost) * self.vec_flow)[0,0]

    def compute_curbcharge_gap(self):
        if self.vec_social_cost is None:
            return None
        return np.dot(np.transpose(self.vec_flow), self.vec_social_cost)[0,0]

    def init_flow(self):
        self.vec_flow = np.array([[0] for x in range(self.num_path)], dtype=float)
        # self._build_const_vec()
        self._eval_cost_function()
        for (u, v) in list(self.demand_graph.edges):
            e = self.demand_graph.edges[(u, v)]
            vec_temp_D = np.array(self.mat_Dpath_between_OD.getrow(e['idx']).transpose().multiply(self.vec_cost).todense())
            vec_temp_R = np.array(self.mat_Rpath_between_OD.getrow(e['idx']).transpose().multiply(self.vec_cost).todense())
            min_D = self._find_min_range(vec_temp_D,self.idx_path_between_OD[u][v][0],self.idx_path_between_OD[u][v][1])
            if self.idx_path_between_OD[u][v][2] != self.idx_path_between_OD[u][v][3]:
                min_R = self._find_min_range(vec_temp_R,self.idx_path_between_OD[u][v][2],self.idx_path_between_OD[u][v][3])
                min = min_D
                if vec_temp_R[min_R,0] < vec_temp_D[min_D,0]:
                    min = min_R
            else:
                min = min_D
            self.vec_flow[min,0] = e['volume']
            
    def init_flow2(self):
        flow_init = np.array([[0] for x in range(self.num_path)], dtype=float)
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            a,b,c,d = self.idx_path_between_OD[u][v]
            init_flow = self.demand_graph.edges[(u, v)]['volume']/(b - a + d - c)
            for i in range(a,b):
                flow_init[i] = init_flow
                # print(i)
            for j in range(c,d):
                flow_init[j] = init_flow
        
        self.vec_flow = flow_init
        
        
    def init_curbcharge(self):
        self.vec_curb_charge = np.array([[self.const_initial_curb_charge] for x in range(self.num_link)])

    def sue_iterate(self,iter,step_size = 1.0,getgap = True):
        self._eval_cost_function()
        vec_change_flow = np.array([[0] for x in range(self.num_path)])
        gap = 0
        if getgap:
            gap = self.compute_sue_gap()
        for (u, v) in list(self.demand_graph.edges):
            # (u, v) = list(self.demand_graph.edges)[0]
            e = self.demand_graph.edges[(u, v)]
            vec_ODpaths_D = self.mat_Dpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths_R = self.mat_Rpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths = vec_ODpaths_D + vec_ODpaths_R
            vec_temp_D = np.array(vec_ODpaths_D.multiply(self.vec_cost).todense())
            vec_temp_R = np.array(vec_ODpaths_R.multiply(self.vec_cost).todense())
            min_D = self._find_min_range(vec_temp_D, self.idx_path_between_OD[u][v][0],self.idx_path_between_OD[u][v][1])
            if self.idx_path_between_OD[u][v][2] != self.idx_path_between_OD[u][v][3]:
                min_R = self._find_min_range(vec_temp_R, self.idx_path_between_OD[u][v][2],self.idx_path_between_OD[u][v][3])
                q_D = (vec_ODpaths_D.transpose() * self.vec_flow)[0,0]
                q_R = (vec_ODpaths_R.transpose() * self.vec_flow)[0,0]
                logit_term_D = (self.const_alpha_D + np.log(q_D)) / self.const_beta
                logit_term_R = (self.const_alpha_R + np.log(q_R)) / self.const_beta
                min_path_idx = 0
                if (self.vec_cost[min_D,0] + logit_term_D <= self.vec_cost[min_R,0] + logit_term_R):
                    min_path_idx = min_D
                else:
                    min_path_idx = min_R
            else:
                min_path_idx = min_D
            
            vec_ODchange = vec_ODpaths.multiply(- step_size / float(iter))
            vec_ODchange[min_path_idx,0] = 0.0
            ODIncrease = -1.0 * (vec_ODchange.transpose() * self.vec_flow)[0,0]
            vec_ODchange = np.array(vec_ODchange.multiply(self.vec_flow).todense())
            vec_ODchange[min_path_idx,0] = ODIncrease
            vec_change_flow = vec_change_flow + vec_ODchange
        
        self.vec_flow = self.vec_flow + vec_change_flow
        self.vec_change_flow = vec_change_flow
        if getgap:
            return gap

    def solve_sue(self, precision = 1e-3, max_iterations = 500, step_size = 0.5, print_gap = True, plot_gap = True):
        self.sue_gap_list = list()
        self.sue_flow_list = list()
        self.sue_cost_list = list()
        self.init_flow()
        self._eval_cost_function()
        initial_gap = self.compute_sue_gap()
        if print_gap:
            print("--- Initilization ---")
            print("Initialized Gap: " + str(initial_gap))
        
        first_score = self.sue_iterate(2,step_size)
        if print_gap:
            print("Iteration 1:")
            print("Gap before flow update: " + str(first_score))
        score = first_score
        self.sue_gap_list.append(score)
        self.sue_flow_list.append(self.vec_flow.transpose().copy())
        self.sue_cost_list.append(self.vec_cost.transpose().copy())
        i = 2
        while score > first_score * precision and i < max_iterations:
            i += 1
            score = self.sue_iterate(i,step_size)
            self.sue_gap_list.append(score)
            self.sue_flow_list.append(self.vec_flow.transpose().copy())
            self.sue_cost_list.append(self.vec_cost.transpose().copy())

            if print_gap:
                print("Iteration " + str(i-1) + ":")
                print("Gap before flow update: " + str(score))
                
        if plot_gap:
            figsize(10, 6)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.plot(range(1,len(self.sue_gap_list)+1), self.sue_gap_list, color='red', linewidth=2)   
            plt.xlabel("iteration",fontsize=16, family = 'arial')
            plt.ylabel("Gap function value",fontsize=16, family = 'arial')
            plt.xlim(0,len(self.sue_gap_list)+1)
            plt.show()
    
    def solve_sue_hcg(self, precision = 1e-3, max_iterations = 500, step_size = 0.5, print_gap = True, plot_gap = True):
        self.sue_gap_list = list()
        self.sue_flow_list = list()
        self.init_flow()
        self._eval_cost_function()
        
        initial_gap = self.compute_sue_gap()
        if print_gap:
            print("--- Initilization ---")
            print("Initialized Gap: " + str(initial_gap))
        
        first_score = self.sue_iterate(2,step_size)
        if print_gap:
            print("Iteration 1:")
            print("Gap before flow update: " + str(first_score))
        score = first_score
        self.sue_gap_list.append(score)
        self.sue_flow_list.append(self.vec_flow.transpose().copy())
        i = 2
        while score > first_score * precision and i < max_iterations:
            i += 1
            self.network_loading()
            path_new, path_new_length, path_new_set, path_new_len = self.new_OD_shortest_path()
            
            if len(path_new) == 0:
                score = self.sue_iterate(i,step_size)
            
            if len(path_new) != 0:
                self.generate_new_OD_shortest_path_vector(path_new, path_new_length, path_new_set, path_new_len)
                score = self.sue_iterate(i,step_size)
            
            self.sue_gap_list.append(score)
            self.sue_flow_list.append(self.vec_flow.transpose().copy())
            if print_gap:
                print("Iteration " + str(i-1) + ":")
                print("Gap before flow update: " + str(score))    
        
        if plot_gap:
            figsize(10, 6)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.plot(range(1,len(self.sue_gap_list)+1), self.sue_gap_list, color='red', linewidth=2)   
            plt.xlabel("iteration",fontsize=16, family = 'arial')
            plt.ylabel("Gap function value",fontsize=16, family = 'arial')
            plt.xlim(0,len(self.sue_gap_list)+1)
            plt.show()
            
    def gradec_curbcharge(self, iteration, step_size, nneg = True):
        self.grad = self._compute_grad_curbcharge()
        # self.grad_sum += self.grad * self.grad
        self.vec_curb_charge -= (step_size/(iteration ** 0.5)) * self.grad
        if nneg:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= 0.0: # add max limit
                    self.vec_curb_charge[i,0] = 0.0
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
        else:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= -self.charge_limit:
                    self.vec_curb_charge[i,0] = -self.charge_limit
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
                    
                    
    def adagrad_curbcharge(self, step_size, nneg = True):
        self.grad = self._compute_grad_curbcharge()
        self.grad_sum += np.multiply(self.grad, self.grad)
        self.vec_curb_charge -= np.multiply(step_size * ((self.grad_sum + 1e-100) ** (-0.5)),self.grad)
        if nneg:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= 0.0:
                    self.vec_curb_charge[i,0] = 0.0
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
        else:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= -self.charge_limit:
                    self.vec_curb_charge[i,0] = -self.charge_limit
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
    
    def normgrad_curbcharge(self, iteration, step_size, nneg = True):
        self.grad = self._compute_grad_curbcharge()
        
        max_grad = self.grad.max()
        min_grad = self.grad.min()
        
        grad_range = max_grad - min_grad
        
        self.grad = self.grad/grad_range
        
        self.vec_curb_charge -= (step_size/(iteration ** 0.5)) * self.grad
        if nneg:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= 0.0:
                    self.vec_curb_charge[i,0] = 0.0
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
        else:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= -self.charge_limit:
                    self.vec_curb_charge[i,0] = -self.charge_limit
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit

    def normgrad_curbcharge2(self, iteration, step_size, nneg = True):
        self.grad = self._compute_grad_curbcharge()
        
        max_grad = self.grad.max()
        min_grad = self.grad.min()
        
        grad_range = max_grad - min_grad
        
        self.grad = self.grad/grad_range
        
        self.vec_curb_charge -= (step_size) * self.grad
        if nneg:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= 0.0:
                    self.vec_curb_charge[i,0] = 0.0
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit
        else:
            for i in range(self.num_link):
                if self.vec_curb_charge[i,0] <= -self.charge_limit:
                    self.vec_curb_charge[i,0] = -self.charge_limit
                # add max limit
                if self.vec_curb_charge[i,0] >= self.charge_limit:
                    self.vec_curb_charge[i,0] = self.charge_limit

    
    def _compute_grad_curbcharge(self):
        # compute path_tilde
        path_index = list(np.where(self.vec_flow > 0)[0])
        
        Delta = self.mat_link_in_path[:,path_index].todense()
        
        Lambda = self.mat_path_between_OD[:,path_index].todense()
        
        link_flow = self.mat_link_in_path * self.vec_flow
        
        vec_demand = np.array([[0] for x in range(len(list(self.demand_graph.edges)))], dtype=float)
        i = 0
        for (u,v) in list(self.demand_graph.edges):
            vec_demand[i,0] = self.demand_graph.edges[(u,v)]['volume']
            i += 1
        
        link_demand = np.concatenate((link_flow, vec_demand), axis = 0)
        
        Delta_Lambda = np.concatenate((Delta, Lambda), axis = 0)
        
        f = np.array([1.0] * len(path_index))
        lb = [0.0] * len(path_index)
        
        mat_A_eq = matlab.double(Delta_Lambda.tolist())
        mat_b_eq = matlab.double(link_demand.tolist())
        mat_f = matlab.double(f.tolist())
        mat_A = matlab.double([])
        mat_b = matlab.double([])
        mat_lb = matlab.double(lb)
        
        eng = matlab.engine.start_matlab()
        
        f_full = eng.linprog(mat_f, mat_A, mat_b, mat_A_eq, mat_b_eq, mat_lb)
        
        eng.quit()
        
        f_full = np.asarray(f_full).flatten()
        
        idx_tilde = list(np.where(f_full > 1e-30)[0])
        
        path_tilde = f_full[idx_tilde].reshape(len(idx_tilde),1)
        
        delta_tilde = Delta[:,idx_tilde]
        lambda_tilde = Lambda[:,idx_tilde]
        # mat_tilde = np.concatenate((delta_tilde, lambda_tilde), axis = 0)
        
        # path_tilde = np.linalg.pinv(mat_tilde).dot(link_demand)
        
        # C_1 constant matrix time cost
        mat_C = np.transpose(self.vec_time_cost)
        
        # M matrix OD_num * path_num
        mat_M = self.mat_path_between_OD
        
        # mat_M  path_num * link_num
        mat_N = self.mat_link_in_path.transpose()
        
        # link_flow
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        
        vec_dPhidx = np.multiply(self.vec_freeflow, np.multiply(np.power(vec_link_flow,3),\
                                                                np.power(self.vec_inv_link_cap,4)) * 0.6)
        mat_dPhidx = np.transpose(vec_dPhidx)
        
        # sp.sparse.diags(vec_dPhidx, [0]).todense()
        
        # dc_dm path_num * link_num
        mat_dcdm = (self.mat_path_start_from_link + self.mat_path_end_at_link).transpose()
        #print("Computing")
        
        # vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
        vec_curb_stops = self.mat_path_end_at_link[:,path_index][:,idx_tilde] * path_tilde + self.mat_path_start_from_link[:,path_index][:,idx_tilde] * path_tilde

        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        inverse_t_temp = (np.array([[1/90] for x in range(self.num_link)], dtype=float))
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], \
                                             self.q_model_rate_threshold)
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] < self.q_model_rate_threshold:
                inverse_t_temp[i, 0] = 0.0
        
        # curb waiting time cost coefficient
        mat_C3 = (np.array([[0] for x in range(self.num_path)], dtype=float)).transpose()
        for j in range(self.num_drive_path, self.num_path):
            mat_C3[0,j] = 0.7
        
        # d omega / d v
        mat_dodv = (np.array([[self.activation_coefficient] for x in range(self.num_link)], dtype=float)).transpose()
        
        # d lambda_1 / d p
        inverse_t = sp.sparse.diags(inverse_t_temp.transpose(),[0])
        
        # d v / d p
        mat_dvdp = np.array([[0.0] for x in range(self.num_link)], dtype=float)
        for i in range(self.num_link):
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] > self.q_model_rate_threshold:
                mat_dvdp[i,0] = self.vec_lambda_2[i, 0]/((self.vec_lambda_2[i, 0] - \
                                                          self.vec_lambda_1[i, 0]) ** 2) * (1/90)
            else:
                mat_dvdp[i,0] = 1/self.q_model_rate_threshold * (1/90)
        
        # d c / d f
        mat_dcdf = sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dPhidx, [0]) * self.mat_link_in_path + \
            sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dodv, [0]) * sp.sparse.diags(mat_dvdp.transpose(), [0]) * \
                (self.mat_path_end_at_link + self.mat_path_start_from_link) + \
            sp.sparse.diags(mat_C3,[0]) * (self.mat_path_end_at_link + self.mat_path_start_from_link).transpose() * sp.sparse.diags(np.transpose(vec_t_curbwait * vec_t_curbwait), [0]) * \
                inverse_t * (self.mat_path_end_at_link + self.mat_path_start_from_link)
        
        mat_dcdf = mat_dcdf[path_index,:][:,path_index][idx_tilde,:][:,idx_tilde]

        mat_util = sp.sparse.bmat([[mat_dcdf, -lambda_tilde.transpose()], [lambda_tilde, None]])
        # mat_preturb = sp.sparse.identity(self.num_path + self.num_demand).multiply(1e-6)
        
        # pseudo inverse of matrix
        mat_util_inv = sp.sparse.linalg.inv(mat_util.transpose() * mat_util) * mat_util.transpose()
        
        mat_util_inv.resize(len(path_tilde), len(path_tilde))
        
        B_11 = mat_util_inv.todense()
        
        df_dm = self.mat_link_in_path.todense().transpose() * delta_tilde * B_11 * mat_dcdm[path_index,:][idx_tilde,:]
        
        gradient = df_dm.transpose() * self.vec_social_cost
        # mat_util = sp.sparse.bmat([[mat_dcdf, -mat_M.transpose()], [mat_M, None]])
        # mat_preturb = sp.sparse.identity(self.num_path + self.num_demand).multiply(1e-6)
        
        # # pseudo inverse of matrix
        # mat_util_inv = sp.sparse.linalg.inv(mat_util.transpose() * mat_util + mat_preturb) * mat_util.transpose()
        
        # mat_util_inv.resize(len(path_tilde), len(path_tilde))
        
        # return - (mat_util_inv * mat_dcdm).transpose() * self.vec_social_cost
        return gradient
    
    
    
    def solve_curbcharge(self, step_size = 1, max_iterations = 15, sue_precision = 1e-3, sue_max_iterations = 3000, \
                         sue_step_size = 0.5, print_gap = True, nneg = True, alg_type = 'adag'):
        self.init_curbcharge() # initialize curbcharge = 0.0
        self.charge_gap_list = list()
        self.flow_list = list()
        self.cost_list = list()
        self.vec_charge = list()
        
        # solve BMUE
        self.solve_sue(sue_precision, sue_max_iterations, sue_step_size, False, False)
        
        gap = self.compute_social_cost(self.vec_flow)
        if print_gap:
            print("Original Gap: " + str(gap))
        self.charge_gap_list.append(gap)
        self.flow_list.append(np.transpose(self.vec_flow.copy()))
        self.cost_list.append(np.transpose(self.vec_cost.copy()))
        self.vec_charge.append(np.transpose(self.vec_curb_charge.copy()))
        
        self.grad_sum = np.array([[0] for x in range(self.num_link)], dtype=float)
        for i in range(max_iterations):
            if print_gap:              
                print("Iteration " + str(i + 1) + ":")
            a = time.time()
            
            if alg_type == 'gd':
                self.gradec_curbcharge(i + 1, step_size, nneg)
                
            if alg_type == 'adag':
                self.adagrad_curbcharge(step_size, nneg)
            
            if alg_type == 'ngd':
                self.normgrad_curbcharge(i + 1, step_size, nneg)

            if alg_type == 'ngd2':
                self.normgrad_curbcharge2(i + 1, step_size, nneg)
            
            # print(self.grad)
            self.solve_sue(sue_precision, sue_max_iterations, sue_step_size, False, False)
            b = time.time()
            path_flow = self.vec_flow
            gap_charge = self.compute_social_cost(path_flow)
            self.charge_gap_list.append(gap_charge)
            self.flow_list.append(np.transpose(self.vec_flow.copy()))
            self.cost_list.append(np.transpose(self.vec_cost.copy()))
            self.vec_charge.append(np.transpose(self.vec_curb_charge.copy()))

            if print_gap:
                
                print("Gap: " + str(gap_charge) + "  time: " + str(round(b - a, 1)))

    
    def compute_auxiliary_flow(self):
        mat_C = np.transpose(self.vec_time_cost)
        # mat_M = self.mat_path_between_OD
        mat_N = self.mat_link_in_path.transpose()
        # mat_N.todense()
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        vec_dPhidx = np.multiply(self.vec_freeflow, np.multiply(np.power(vec_link_flow,3),\
                                                                np.power(self.vec_inv_link_cap,4)) * 0.6)
        mat_dPhidx = np.transpose(vec_dPhidx)
        
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
            
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        inverse_t_temp = (np.array([[1/90] for x in range(self.num_link)], dtype=float))
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], \
                                             self.q_model_rate_threshold)
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] < self.q_model_rate_threshold:
                inverse_t_temp[i, 0] = 0.0
        
        mat_C3 = (np.array([[0] for x in range(self.num_path)], dtype=float)).transpose()
        for j in range(self.num_drive_path, self.num_path):
            mat_C3[0,j] = 0.7
        
        mat_dodv = (np.array([[self.activation_coefficient] for x in range(self.num_link)], dtype=float)).transpose()
            
        inverse_t = sp.sparse.diags(inverse_t_temp.transpose(),[0])
        
        mat_dvdp = np.array([[0.0] for x in range(self.num_link)], dtype=float)
        for i in range(self.num_link):
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] > self.q_model_rate_threshold:
                mat_dvdp[i,0] = self.vec_lambda_2[i, 0]/((self.vec_lambda_2[i, 0] - \
                                                          self.vec_lambda_1[i, 0]) ** 2) * (1/90)
            else:
                mat_dvdp[i,0] = 1/self.q_model_rate_threshold * (1/90)
        
        mat_dcdf = sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dPhidx, [0]) * self.mat_link_in_path + \
            sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dodv, [0]) * \
                sp.sparse.diags(mat_dvdp.transpose(), [0]) * \
                (self.mat_path_end_at_link + self.mat_path_start_from_link) + \
            sp.sparse.diags(mat_C3,[0]) * (self.mat_path_end_at_link + self.mat_path_start_from_link).transpose() * \
                sp.sparse.diags(np.transpose(vec_t_curbwait * vec_t_curbwait), [0]) * \
                inverse_t * (self.mat_path_end_at_link + self.mat_path_start_from_link)
                    
        grad_flow = np.transpose(self.vec_flow) * mat_dcdf
        grad_flow = grad_flow[0]
        sc = self.vec_social_cost.transpose()[0]
        grad_flow += sc

        # find path with least gradient value
        self.auxiliary_flow = np.array([[0] for x in range(self.num_path)], dtype=float)
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            e = self.demand_graph.edges[(u, v)]
            min_D = self._find_min_so(grad_flow, self.idx_path_between_OD[u][v][0],self.idx_path_between_OD[u][v][1])
            
            if self.idx_path_between_OD[u][v][2] != self.idx_path_between_OD[u][v][3]:
                min_R = self._find_min_so(grad_flow, self.idx_path_between_OD[u][v][2],self.idx_path_between_OD[u][v][3])
                min = min_D
                if grad_flow[min_R] < grad_flow[min_D]:
                    min = min_R
            else:
                min = min_D
            
            self.auxiliary_flow[min,0] = e['volume']
        
    def compute_so_gap(self):       
        vec_min_grad = np.array([[0] for x in range(self.num_path)], dtype = float)
        mat_C = np.transpose(self.vec_time_cost)
        # mat_M = self.mat_path_between_OD
        mat_N = self.mat_link_in_path.transpose()
        # mat_N.todense()
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        vec_dPhidx = np.multiply(self.vec_freeflow, np.multiply(np.power(vec_link_flow,3),\
                                                                np.power(self.vec_inv_link_cap,4)) * 0.6)
        mat_dPhidx = np.transpose(vec_dPhidx)
        
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
            
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        inverse_t_temp = (np.array([[1/90] for x in range(self.num_link)], dtype=float))
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], \
                                             self.q_model_rate_threshold)
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] < self.q_model_rate_threshold:
                inverse_t_temp[i, 0] = 0.0
        
        mat_C3 = (np.array([[0] for x in range(self.num_path)], dtype=float)).transpose()
        for j in range(self.num_drive_path, self.num_path):
            mat_C3[0,j] = 0.7
        
        mat_dodv = (np.array([[self.activation_coefficient] for x in range(self.num_link)], dtype=float)).transpose()
            
        inverse_t = sp.sparse.diags(inverse_t_temp.transpose(),[0])
        
        mat_dvdp = np.array([[0.0] for x in range(self.num_link)], dtype=float)
        for i in range(self.num_link):
            if self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0] > self.q_model_rate_threshold:
                mat_dvdp[i,0] = self.vec_lambda_2[i, 0]/((self.vec_lambda_2[i, 0] - \
                                                          self.vec_lambda_1[i, 0]) ** 2) * (1/90)
            else:
                mat_dvdp[i,0] = 1/self.q_model_rate_threshold * (1/90)
        
        mat_dcdf = sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dPhidx, [0]) * self.mat_link_in_path + \
            sp.sparse.diags(mat_C,[0]) * mat_N * sp.sparse.diags(mat_dodv, [0]) * \
                sp.sparse.diags(mat_dvdp.transpose(), [0]) * \
                (self.mat_path_end_at_link + self.mat_path_start_from_link) + \
            sp.sparse.diags(mat_C3,[0]) * mat_N * sp.sparse.diags(np.transpose(vec_t_curbwait * vec_t_curbwait), [0]) * \
                inverse_t * (self.mat_path_end_at_link + self.mat_path_start_from_link)
        
        grad_flow = np.transpose(self.vec_flow) * mat_dcdf
        # grad_flow = grad_flow[0]
        # sc = self.vec_social_cost.transpose()[0]
        sc = self.vec_social_cost.transpose()
        grad_flow += sc
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            e = self.demand_graph.edges[(u, v)]
            vec_ODpaths_D = self.mat_Dpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths_R = self.mat_Rpath_between_OD.getrow(e['idx']).transpose()
            vec_ODpaths = vec_ODpaths_D + vec_ODpaths_R
            min_D = self._find_min_so(grad_flow[0], self.idx_path_between_OD[u][v][0],self.idx_path_between_OD[u][v][1])
            if self.idx_path_between_OD[u][v][2] != self.idx_path_between_OD[u][v][3]:
                min_R = self._find_min_so(grad_flow[0], self.idx_path_between_OD[u][v][2],self.idx_path_between_OD[u][v][3])
                min_grad = grad_flow[0,min_D]
                if grad_flow[0,min_R] < grad_flow[0,min_D]:
                    min_grad = grad_flow[0,min_R]
            else:
                min_grad = grad_flow[0,min_D]
            vec_min_grad += vec_ODpaths.multiply(min_grad)
        
        return ((grad_flow - np.transpose(vec_min_grad)) * self.vec_flow)[0,0]/(np.transpose(vec_min_grad) * self.vec_flow)[0,0]
        
    def solve_so(self, flow_init, max_iterations = 1000, plot_gap = True):
        # initialization
        self.so_gap_list = list()
        self.so_sc_list = list()
        self.vec_flow = flow_init
        self._eval_cost_function()
        
        nu = 0
        lambda_nu = 1/(1 + nu)
        while (nu < max_iterations):            
            self.compute_auxiliary_flow()
            self.vec_flow = (1 - lambda_nu) * self.vec_flow + lambda_nu * self.auxiliary_flow
            self._eval_cost_function()
            score = self.compute_so_gap()
            self.so_gap_list.append(score)
            total_sc = self.compute_social_cost(self.vec_flow)
            self.so_sc_list.append(total_sc)
            nu += 1
            lambda_nu = 1/(1 + nu)
            if plot_gap:
                print("iteration = " + str(nu) + "   gap = " + str(score))
    
    def _compute_lambdas(self):
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)    
     
        
    def _eval_cost_function(self):
        self.vec_time_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        vec_const = np.array([[0] for x in range(self.num_path)], dtype=float)
        vec_distance_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        for i in range(self.num_drive_path):
            vec_distance_cost[i, 0] = self.const_driving_cost # The per mile unit cost constant for driving = 1.5
            vec_const[i, 0] = self.const_dest_parking # destination parking price = 20
            self.vec_time_cost[i, 0] = self.const_vtime # Time cost constant
        for i in range(self.num_drive_path, self.num_path):
            vec_distance_cost[i, 0] = self.const_rhfare_dist # 1.75
            vec_const[i, 0] = self.const_rhfare_overhead # 2.55
            self.vec_time_cost[i, 0] = self.const_vtime + self.const_rhfare_time
        vec_walk_time = self.vec_walking_dist * (1.0 / self.const_walking_speed)
        self.vec_const_cost = vec_const + np.multiply(vec_distance_cost, self.vec_len_paths) + vec_walk_time * self.const_vtime
        
        # lambda
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        vec_t_queue = np.array([[0.0] for x in range(self.num_link)])
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], \
                                             self.q_model_rate_threshold) # w_a
            vec_t_queue[i,0] = self.vec_lambda_1[i, 0] / max(self.vec_lambda_2[i, 0] - \
                                                             self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # v_a
        vec_temp_const = np.array([[1.0] for x in range(self.num_link)])
        
        vec_temp_flow_penalty = np.multiply(np.power(vec_link_flow,4),np.power(self.vec_inv_link_cap,4) * 0.15) # bug
        
        vec_link_time = np.multiply(self.vec_freeflow, vec_temp_flow_penalty + vec_temp_const)
        
        # add queue impact
        vec_path_time = self.mat_link_in_path.transpose() * (vec_link_time + self.activation_coefficient * vec_t_queue)
        
        # fix path travel time for driving and RH
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            (dr_b, dr_e, rh_b, rh_e) = self.idx_path_between_OD[u][v]
            for i in list(range(dr_b, dr_e)):
                # i = list(range(dr_b, dr_e))[0]
                path_temp = self.paths[i]
                if not self.modified_graph.nodes[path_temp[0]].get('isOriginal', True):
                    
                    origin_walk_dis = self.demand_graph.nodes[u]['neighborhood'][path_temp[0]]
                    
                    origin_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (origin_walk_dis/origin_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                if not self.modified_graph.nodes[path_temp[-1]].get('isOriginal', True):
                    destination_walk_dis = self.demand_graph.nodes[v]['neighborhood'][path_temp[-1]]
                    
                    destination_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (destination_walk_dis/destination_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
            
            for i in list(range(rh_b, rh_e)):
                # i = list(range(rh_b, rh_e))[0]
                path_temp = self.paths[i]
                if not self.modified_graph.nodes[path_temp[0]].get('isOriginal', True):
                    
                    origin_walk_dis = self.demand_graph.nodes[u]['neighborhood'][path_temp[0]]
                    
                    origin_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (origin_walk_dis/origin_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                if not self.modified_graph.nodes[path_temp[-1]].get('isOriginal', True):
                    destination_walk_dis = self.demand_graph.nodes[v]['neighborhood'][path_temp[-1]]
                    
                    destination_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (destination_walk_dis/destination_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
                    
        # path flow cost
        self.vec_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        self.vec_cost = np.multiply(vec_path_time, self.vec_time_cost) + self.vec_const_cost
        
        # add path curb stop cost
        self.vec_cost += np.multiply((self.mat_path_end_at_link + self.mat_path_start_from_link).transpose() * \
                                     vec_t_curbwait, self.const_vtime)
        
        # total social cost
        self.vec_social_cost = self.vec_cost.copy()
        
        # add curb charge
        self.vec_cost += (self.mat_path_start_from_link + self.mat_path_end_at_link).transpose() * self.vec_curb_charge
        
    def compute_social_cost(self, vec_flow):
        vec_const = np.array([[0] for x in range(self.num_path)], dtype=float)
        vec_distance_cost = np.array([[0] for x in range(self.num_path)], dtype=float)
        for i in range(self.num_drive_path):
            vec_distance_cost[i, 0] = self.const_driving_cost # The per mile unit cost constant for driving = 1.5
            vec_const[i, 0] = self.const_dest_parking # destination parking price = 20
            self.vec_time_cost[i, 0] = self.const_vtime # Time cost constant
        for i in range(self.num_drive_path, self.num_path):
            vec_distance_cost[i, 0] = self.const_rhfare_dist # 1.75
            vec_const[i, 0] = self.const_rhfare_overhead # 2.55
            self.vec_time_cost[i, 0] = self.const_vtime + self.const_rhfare_time
        vec_walk_time = self.vec_walking_dist * (1.0 / self.const_walking_speed)
        self.vec_const_cost = vec_const + np.multiply(vec_distance_cost, self.vec_len_paths) + vec_walk_time * self.const_vtime
        
        # lambda
        vec_curb_stops = self.mat_path_end_at_link * vec_flow + self.mat_path_start_from_link * vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        
        vec_link_flow = self.mat_link_in_path * vec_flow
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        vec_t_queue = np.array([[0.0] for x in range(self.num_link)])
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], \
                                             self.q_model_rate_threshold) # w_a
            vec_t_queue[i,0] = self.vec_lambda_1[i, 0] / max(self.vec_lambda_2[i, 0] - \
                                                             self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # v_a
        vec_temp_const = np.array([[1.0] for x in range(self.num_link)])
        
        vec_temp_flow_penalty = np.multiply(np.power(vec_link_flow,4),np.power(self.vec_inv_link_cap,4) * 0.15) # bug
        
        vec_link_time = np.multiply(self.vec_freeflow, vec_temp_flow_penalty + vec_temp_const)
        
        # add queue impact
        vec_path_time = self.mat_link_in_path.transpose() * (vec_link_time + self.activation_coefficient * vec_t_queue)
        
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            (dr_b, dr_e, rh_b, rh_e) = self.idx_path_between_OD[u][v]
            for i in list(range(dr_b, dr_e)):
                # i = list(range(dr_b, dr_e))[0]
                path_temp = self.paths[i]
                if not self.modified_graph.nodes[path_temp[0]].get('isOriginal', True):
                    
                    origin_walk_dis = self.demand_graph.nodes[u]['neighborhood'][path_temp[0]]
                    
                    origin_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (origin_walk_dis/origin_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                if not self.modified_graph.nodes[path_temp[-1]].get('isOriginal', True):
                    destination_walk_dis = self.demand_graph.nodes[v]['neighborhood'][path_temp[-1]]
                    
                    destination_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (destination_walk_dis/destination_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
            
            for i in list(range(rh_b, rh_e)):
                # i = list(range(rh_b, rh_e))[0]
                path_temp = self.paths[i]
                if not self.modified_graph.nodes[path_temp[0]].get('isOriginal', True):
                    
                    origin_walk_dis = self.demand_graph.nodes[u]['neighborhood'][path_temp[0]]
                    
                    origin_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (origin_walk_dis/origin_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[0]]['edgeidx'],0]
                    
                if not self.modified_graph.nodes[path_temp[-1]].get('isOriginal', True):
                    destination_walk_dis = self.demand_graph.nodes[v]['neighborhood'][path_temp[-1]]
                    
                    destination_link_length = self.vec_link_len[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
                    
                    vec_path_time[i,0] = vec_path_time[i,0] - (destination_walk_dis/destination_link_length) * vec_link_time[self.modified_graph.nodes[path_temp[-1]]['edgeidx'],0]
        
        # path flow cost
        vec_cost = np.multiply(vec_path_time, self.vec_time_cost) + self.vec_const_cost
        # add path curb stop cost
        vec_cost += np.multiply((self.mat_path_end_at_link + self.mat_path_start_from_link).transpose() * \
                                vec_t_curbwait, self.const_vtime)
        # total social cost
        # vec_social_cost = vec_cost.copy()
        social_cost = np.dot(np.transpose(vec_flow), vec_cost)[0,0]
        return social_cost
    
    def compute_revenue_parking(self, vec_flow):  
        # vec_flow = self.vec_flow
        vec_flow_driving = vec_flow[0:(self.num_drive_path)]
        return sum(vec_flow_driving)[0] * self.const_dest_parking
    
    def compute_revenue_curb(self, vec_flow, vec_curb_charge):
        path_curb_cost = (self.mat_path_start_from_link + self.mat_path_end_at_link).transpose() * vec_curb_charge
        return sum(vec_flow * path_curb_cost)[0]
    
    def compute_curb_congestion_cost(self, vec_flow):
        vec_link_flow = self.mat_link_in_path * vec_flow
        
        # curb impact
        vec_curb_stops = self.mat_path_end_at_link * vec_flow + self.mat_path_start_from_link * vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        vec_t_queue = np.array([[0.0] for x in range(self.num_link)])
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # w_a
            vec_t_queue[i,0] = self.vec_lambda_1[i, 0] / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # v_a

        return sum(self.activation_coefficient * vec_t_queue + self.const_vtime * vec_t_curbwait)
    
    def network_loading(self):
        # vec_flow is path_flow
        # convert path flow to link flow
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        
        # curb impact
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        vec_t_curbwait = np.array([[0.0] for x in range(self.num_link)])
        vec_t_queue = np.array([[0.0] for x in range(self.num_link)])
        for i in range(self.num_link):
            vec_t_curbwait[i, 0] = 1.0 / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # w_a
            vec_t_queue[i,0] = self.vec_lambda_1[i, 0] / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # v_a
        
        vec_link_travel_time = np.multiply(self.vec_freeflow, np.array([[1.0] for x in range(self.num_link)]) + \
                                           np.multiply(np.power(vec_link_flow,4),np.power(self.vec_inv_link_cap,4) * 0.15))
        
        vec_link_travel_time += self.activation_coefficient * vec_t_queue
        
        # update modified graph link cost
        for (u, v) in list(self.modified_graph.edges):
            e = self.modified_graph.edges[(u,v)]
            fft = self.vec_freeflow[e['index'],0]
            link_flow = vec_link_flow[e['index'],0]
            inv_capacity = self.vec_inv_link_cap[e['index'],0]
            self.modified_graph.edges[(u,v)]['tt'] = fft * (1 + 0.15 * np.power(link_flow * inv_capacity, 4)) + 0.05 * vec_t_queue[e['index'],0]
    
    def new_OD_shortest_path(self):
        vec_link_flow = self.mat_link_in_path * self.vec_flow
        
        # curb impact
        vec_curb_stops = self.mat_path_end_at_link * self.vec_flow + self.mat_path_start_from_link * self.vec_flow
        self.vec_lambda_1 = vec_curb_stops * (1.0 / self.const_t_commute)
        self.vec_lambda_2 = np.multiply(self.vec_link_len, self.vec_curb_cap) * (1.0/self.const_t_curbstop)
        vec_t_queue = np.array([[0.0] for x in range(self.num_link)])
        
        for i in range(self.num_link):
            vec_t_queue[i,0] = self.vec_lambda_1[i, 0] / max(self.vec_lambda_2[i, 0] - self.vec_lambda_1[i, 0], self.q_model_rate_threshold) # v_a
        
        vec_link_travel_time = np.multiply(self.vec_freeflow, np.array([[1.0] for x in range(self.num_link)]) + \
                                           np.multiply(np.power(vec_link_flow,4),np.power(self.vec_inv_link_cap,4) * 0.15))
        
        vec_link_travel_time += self.activation_coefficient * vec_t_queue
        
        path_travel_time = self.mat_link_in_path.transpose() * vec_link_travel_time
        
        path_new = dict()
        path_new_length = dict()
        path_new_set = dict()
        
        for (u,v) in list(self.demand_graph.edges):
            # (u,v) = list(self.demand_graph.edges)[0]
            # print (u,v)
            origin_set = list(self.demand_graph.nodes[u]['neighborhood'].keys())
            destination_set = list(self.demand_graph.nodes[v]['neighborhood'].keys())
            origin_set.append(u)
            destination_set.append(v)
            shortest_paths_set = []
            shortest_path_length_set = []
            for o in origin_set:
                for d in destination_set:
                    if nx.has_path(self.modified_graph,o,d):
                        new_path = nx.shortest_path(self.modified_graph, source=o, target=d, weight='tt', method='dijkstra')
                        
                        # print(nx.info(self.modified_graph))
                        
                        # self.modified_graph.nodes
                        # # new_path = path_new[(3093,7551)]                        
                        # new_path_length = 0.0
                        # for n in range(len(new_path)):
                        #     if n != len(new_path) - 1:
                        #         new_path_length += self.modified_graph.edges[(new_path[11], new_path[12])]['length']
                                
                        #         print(n)
                        new_path_length = nx.shortest_path_length(self.modified_graph, source=o, target=d, weight='tt', method='dijkstra')
                        
                        
                        
                        # adding element in dict
                        shortest_paths_set.append(new_path)
                        shortest_path_length_set.append(new_path_length)
            new_path_final = shortest_paths_set[shortest_path_length_set.index(min(shortest_path_length_set))]
            new_path_length_final = min(shortest_path_length_set)
            
            # check driving or ridehailing
            if new_path_final[0] == u:
                new_path_set_final = 'DR'
            if new_path_final[0] != u and new_path_final[-1] != v:
                new_path_set_final = 'RH'
            
            # check if it is shorter than current paths
            a_,b_,c_,d_ = self.idx_path_between_OD[u][v]
            list_index = list(range(a_,b_)) + list(range(c_,d_))
            path_compare = path_travel_time[list_index]
            if min(path_compare)[0] > new_path_length_final:
                path_new[(u,v)] = new_path_final
                path_new_length[(u,v)] = new_path_length_final
                path_new_set[(u,v)] = new_path_set_final
            
        path_new_len = dict()
        for (u,v) in list(path_new.keys()):
            path_temp = path_new[(u,v)]
            path_length_temp = 0.0
            for n in range(0, len(path_temp)-1):
                path_length_temp += self.modified_graph.edges[(path_temp[n], path_temp[n+1])]['length']
            path_new_len[(u,v)] = path_length_temp
                        
        return path_new, path_new_length, path_new_set, path_new_len
    
    def generate_new_OD_shortest_path_vector(self, path_new, path_new_length, path_new_set, path_new_len):
        num_link = len(self.network_graph.edges)
        num_path = len(self.paths_raw) + len(path_new.keys())
        
        num_demand = len(self.demand_graph.edges)
        
        link_in_path = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        path_start_from_link = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        path_end_at_link = sp.sparse.dok_matrix((num_link,num_path), dtype = np.float32)
        Dpath_between_OD = sp.sparse.dok_matrix((num_demand,num_path), dtype = np.float32)
        Rpath_between_OD = sp.sparse.dok_matrix((num_demand, num_path), dtype=np.float32)
        ODvector = sp.sparse.dok_matrix((num_demand,1), dtype = np.float32)
        vec_walking_dist = sp.sparse.dok_matrix((num_path,1), dtype = np.float32)
        
        
        idx_OD = 0
        count_new_dr = 0
        count_new_rh = 0
        
        for O in self.idx_path_between_OD.keys():
            for D in self.idx_path_between_OD[O].keys():
                (idx_db,idx_de,idx_rb,idx_re) = self.idx_path_between_OD[O][D]
                
                idx_db += count_new_dr
                idx_de += count_new_dr
                idx_rb += count_new_dr + count_new_rh
                idx_re += count_new_dr + count_new_rh
                
                if (O,D) in path_new:
                    if path_new_set[(O,D)] == 'DR':
                        count_new_dr += 1
                        self.paths_raw.insert(idx_de, path_new[(O,D)])
                        self.vec_flow = np.insert(self.vec_flow, idx_de, values = np.array([[0.0]]), axis=0)
                        self.vec_len_paths = np.insert(self.vec_len_paths, idx_de, values = np.array([[path_new_len[(O,D)]]]), axis=0)
                        self.idx_path_between_OD[O][D] = (idx_db,idx_de + 1,idx_rb,idx_re)
                        # print(self.idx_path_between_OD[O][D])
                    
                    if path_new_set[(O,D)] == 'RH': 
                        count_new_rh += 1
                        self.paths_raw.insert(idx_re, path_new[(O,D)])
                        self.vec_flow = np.insert(self.vec_flow, idx_re, values=np.array([[0.0]]), axis=0)
                        self.vec_len_paths = np.insert(self.vec_len_paths, idx_re, values = np.array([[path_new_len[(O,D)]]]), axis=0)
                        self.idx_path_between_OD[O][D] = (idx_db,idx_de,idx_rb,idx_re + 1)
                        # print(self.idx_path_between_OD[O][D])
                else:
                    self.idx_path_between_OD[O][D] = (idx_db,idx_de,idx_rb,idx_re)
        
        # reverse path and delete the curb node in the middle of paths
        self.paths = []
        for i in range(num_path):
            # i = 0
            path = self.paths_raw[i]
            node_remove = []
            
            if not self.modified_graph.nodes[path[0]].get('isOriginal', True):
                link_in_path[self.modified_graph.nodes[path[0]]['edgeidx'],i] = 1
                             
                if (i > self.num_drive_path + sum(value == 'DR' for value in path_new_set.values()))|(i == self.num_drive_path + sum(value == 'DR' for value in path_new_set.values())):
                    path_start_from_link[self.modified_graph.nodes[path[0]]['edgeidx'],i] = 1
                
                
            if not self.modified_graph.nodes[path[len(path)-1]].get('isOriginal', True):
                link_in_path[self.modified_graph.nodes[path[len(path)-1]]['edgeidx'], i] = 1
                if (i > self.num_drive_path + sum(value == 'DR' for value in path_new_set.values()))|(i == self.num_drive_path + sum(value == 'DR' for value in path_new_set.values())):
                    path_end_at_link[self.modified_graph.nodes[path[len(path)-1]]['edgeidx'], i] = 1
                
            for j in range(len(path)-2,0,-1):
                k = path[j]
                if not self.modified_graph.nodes[k].get('isOriginal', True):
                    link_in_path[self.modified_graph.nodes[k]['edgeidx'], i] = 1
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
                        dis += self.demand_graph.nodes[O]['neighborhood'][path[0]]
                    if path[-1] != D:
                        dis += self.demand_graph.nodes[D]['neighborhood'][path[-1]]
                    if (path[0] != O)|(path[-1] != D):
                        vec_walking_dist[i,0] = dis
                        
                for i in range(idx_rb, idx_re):
                    Rpath_between_OD[idx_OD,i] = 1
                    path = self.paths[i]
                    vec_walking_dist[i,0] = self.demand_graph.nodes[O]['neighborhood'][path[0]] + self.demand_graph.nodes[D]['neighborhood'][path[-1]]
                    
                ODvector[idx_OD,0] = self.demand_graph.edges[(O,D)]['volume']
                
                self.demand_graph.edges[(O,D)]['idx'] = idx_OD
                
                idx_OD += 1
        
        self.mat_Dpath_between_OD = Dpath_between_OD.tocsr()
        self.mat_Rpath_between_OD = Rpath_between_OD.tocsr()
        self.mat_path_between_OD = Dpath_between_OD + Rpath_between_OD
        self.mat_link_in_path = link_in_path.tocsr()
        self.mat_path_end_at_link = path_end_at_link.tocsr()
        self.mat_path_start_from_link = path_start_from_link.tocsr()
        self.vec_OD = ODvector.tocsr()
        self.vec_walking_dist = vec_walking_dist
        self.num_path = len(self.paths)
        self.num_drive_path = self.num_drive_path + sum(value == 'DR' for value in path_new_set.values())
        self.num_ridehailing_path = self.num_ridehailing_path + sum(value == 'RH' for value in path_new_set.values())
                
    
    
    
    