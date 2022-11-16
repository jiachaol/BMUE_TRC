import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

class fn_and_const():
    def __init__(self):
        self.const_vtime = 0.1 # $/min
        self.const_driving_cost = 0.5 # $/min
        self.const_dest_parking = 20.0
        self.const_rhfare_time = 0.35
        self.const_rhfare_dist = 1.75
        self.const_rhfare_overhead = 2.55
        self.const_walking_speed = 0.05 # mile/min
        self.const_curb_cap = 50
        self.const_t_commute = 90
        self.const_t_curbstop = 2
        self.const_alpha_D = 1
        self.const_alpha_R = 2
        self.const_beta = 1.0
        self.q_model_rate_threshold = 1e-2
        self.const_initial_curb_charge = 0.0
        self.charge_limit = 20
        self.activation_coefficient = 0.05

    def fn_activation(self,a,b):
        return 0.0

    def fn_d_activation(self,a,b):
        return 0.0