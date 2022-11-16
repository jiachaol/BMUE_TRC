import os
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial

# GLB_name_list = ['Anaheim', 'Austin', 'Barcelona', 'Berlin-Center', 'Berlin-Friedrichshain',
#                   'Berlin-Mitte-Center', 'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center',
#                   'Berlin-Prenzlauerberg-Center', 'Berlin-Tiergarten', 'Birmingham-England',
#                   'Braess-Example', 'chicago-regional', 'Chicago-Sketch', 'Eastern-Massachusetts',
#                   'Hessen-Asymmetric', 'Philadelphia', 'SiouxFalls', 'SymmetricaTestCase',
#                   'Terrassa-Asymmetric', 'Winnipeg', 'Winnipeg-Asymmetric']

# GLB_name_list = ['SiouxFalls', 'chicago-regional', 'Austin', 'Anaheim', 'Barcelona',
#                 'Berlin-Center', 'Birmingham-England', 'Philadelphia', 'Winnipeg', 'Tester',
#                 'Pittsburgh', 'pgh']


class network_handler():
  def __init__(self, name, tn_folder):
    # assert(name in GLB_name_list)
    self.name = name
    self.folder = os.path.join(tn_folder, name)
    self.net_df = None
    self.od_dict = None
    self.network_graph = None
    self.demand_graph = None
    self.max_node = 0;
    self.max_edge = 0;
    self._read_folder()


  def _read_folder(self):
    self._read_net()
    self._read_od()


  def _read_net(self):
    file_name = find_file('net', self.folder)
    # print(pd.read_table(os.path.join(self.folder, file_name),
    #                   comment= '<', skipinitialspace= True, delim_whitespace = True).dropna(axis = 1).columns)
    self.net_df = pd.read_csv(os.path.join(self.folder, file_name),
                      comment= '<', skipinitialspace= True, delim_whitespace = True).dropna(axis = 1).drop(labels = ';', axis = 1)

    
  def build_network_graph(self, unit_cov = 1.0, curb = 0.5):
    G = nx.DiGraph()
    for row in self.net_df.iterrows():
        start_node = np.int(row[1][0])
        end_node = np.int(row[1][1])
        cap = np.float(row[1][2])
        l = np.float(row[1][3])
        fft = np.float(row[1][4])
        B = np.float(row[1][5])
        Power = np.float(row[1][6])
        if len(row[1]) >= 11:
            curb = np.float(row[1][10])
        self.max_node = max(self.max_node, start_node, end_node)
        G.add_edge(start_node, end_node, cap = cap, length = l * unit_cov,
                   fft = fft, B = B, power = Power, index = self.max_edge, curb = curb)
        self.max_edge += 1
    self.max_node += 1
    self.network_graph = G


  def build_demand_graph(self):
    demand_graph = nx.DiGraph()
    for O in self.od_dict.keys():
      assert(O in self.network_graph.nodes())
      for D in self.od_dict[O].keys():
        assert(D in self.network_graph.nodes())
        if O == D:
          continue
        total_demand = self.od_dict[O][D]
        demand_graph.add_edge(O, D, volume = total_demand)
    self.demand_graph = demand_graph


  def _read_od(self):
    file_name = find_file('trips', self.folder)
    f = open(os.path.join(self.folder, file_name), 'r')
    line = f.readline()
    demand_dict = dict()
    while(line):
        # print ('s',line, line.strip('\n').strip('\r'))
        if line.startswith("<") or line.strip('\n').strip('\r').strip('\t') == "":
            line = f.readline()
            continue
        if line.startswith("Origin"):
            O = int(line.split()[1])
            # print (O)
            demand_dict[O] = dict()
            line = f.readline()
            # print (line)
            while(len(line.strip('\n').split(";")) > 0):
                if line.startswith("<") or line.strip('\n').strip('\r').strip('\t') == "":
                    line = f.readline()
                    break
                if line.startswith("Origin"):
                    break
                words = line.strip().split(";")[:-1]
                for word in words:
                    (D, demand) = word.split(':')
                    if np.float(demand) > 0:
                      demand_dict[O][int(D)] = np.float(demand)
                line = f.readline()
    self.od_dict = demand_dict
    f.close()


  # def get_total_demand(self):
  #   tot_OD = 0.0
  #   for O in self.od_dict.keys():
  #     for D in self.od_dict[O].keys():
  #       tot_OD += self.od_dict[O][D]
  #   return tot_OD


  def check(self):
    assert(self.net_df is not None)
    assert(self.od_dict is not None)
    # for O in self.od_dict.keys():
    #   for D in self.od_dict.keys():
    #     assert(O )


def find_file(s, folder):
  file_list = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
  # print (file_list)
  tmp_list = list(filter(lambda x: s in x and '.tntp' in x, file_list))
  # print (tmp_list)
  assert(len(tmp_list) == 1)
  return tmp_list[0]


def create_BPR_function(fft, B, cap, power):
 # free flow time * ( 1 + B * (flow/capacity)^Power )
  bpr = lambda fft, B, cap, power, x: fft * (1 + B * np.power(x/cap, power))
  dbpr = lambda fft, B, cap, power, x: fft * B * power * np.power(x/cap, power-1) * (1/cap)
  partial_brp = partial(bpr, fft, B, cap, power)
  partial_dbrp = partial(dbpr, fft, B, cap, power)
  return partial_brp, partial_dbrp



