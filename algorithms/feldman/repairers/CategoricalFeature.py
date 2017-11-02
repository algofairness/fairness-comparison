import networkx as nx
from collections import defaultdict
import random

class CategoricalFeature:
  def __init__(self, data, name="no_name"):
    self.data = data
    self.name = name
    d1=defaultdict(int) #bin_data
    d2=defaultdict(int) #bin_index_dict
    d3=defaultdict(list) #bin_fulldata
    d4=defaultdict(int) #bin_index_dict_reverse
    d5=defaultdict(int) #category_count

    n = len(self.data)
    count = 0
    for i in range(0,n):
      obs = self.data[i]
      if obs in d2: pass  # if obs (i.e. category) is alreay a KEY in bin_index_data then don't do anything
      else:
        d2[obs] = count #bin_index_dict inits the KEY: category, with VALUE: count
        d4[count] = obs #bin_index_dict_reverse does the opposite
        count += 1
      bin_idx = d2[obs]
      d1[bin_idx] += 1 #add 1 to the obs category idex in bin_data
      d5[obs] += 1 #add 1 to the obs category NAME in category_count
      d3[bin_idx].append(i) #add obs to the list of obs with that category in bin_fulldata

    self.bin_data = d1
    self.category_count = d5
    self.num_bins = len(d1.items())
    self.bin_fulldata = d3
    self.bin_index_dict = d2
    self.bin_index_dict_reverse = d4


  def create_graph(self, count_generator): #creates graph given a CategoricalFeature object
    DG=nx.DiGraph() #using networkx package
    bin_list = self.bin_data.items()
    bin_index_dict_reverse = self.bin_index_dict_reverse
    k = self.num_bins
    DG.add_node('s')
    DG.add_node('t')
    for i in range(0, k): #lefthand side nodes have capacity = number of observations in category i
      DG.add_node(i)
      DG.add_edge('s', i, {'capacity' : bin_list[i][1], 'weight' : 0})
    for i in range(k, 2*k): #righthand side nodes have capacity = DESIRED number of observations in category i
      DG.add_node(i)
      cat = bin_index_dict_reverse[i-k]
      desired_count = count_generator(cat)
      DG.add_edge(i, 't', {'capacity' : desired_count, 'weight' : 0})
    #Add special node to hold overflow
    DG.add_node(2*k)
    DG.add_edge(2*k, 't', {'weight' : 0})
    for i in range(0, k):
      for j in range(k,2*k): #for each edge from a lefthand side node to a righhand side node:
        if (i+k)==j:  #IF they represent the same category, the edge weight is 0
          DG.add_edge(i, j, {'weight' : 0})
        else: #IF they represent different categories, the edge weight is 1
          DG.add_edge(i, j, {'weight' : 1})
      #THIS IS THE OVERFLOW NODE!!
      DG.add_edge(i, 2*k, {'weight' : 2})
    return DG

  def repair(self, DG): #new_feature = repair_feature(feature, create_graph(feature))
    mincostFlow = nx.max_flow_min_cost(DG, 's', 't') #max_flow_min_cost returns Dictionary of dictionaries. Keyed by nodes such that mincostFlow[u][v] is the flow edge (u,v)
    bin_dict = self.bin_fulldata
    index_dict = self.bin_index_dict_reverse
    size_data = len(self.data)
    repair_bin_dict = {}
    repair_data = [0]*size_data #initialize repaired data to be 0. If there are zero's after we fill it in the those observations belong in the overflow, "no category"
    k = self.num_bins
    overflow = 0
    for i in range(0,k): #for each lefthand side node i
      overflow += mincostFlow[i][2*k]
      for j in range(k, 2*k): #for each righthand side node j
        edgeflow = mincostFlow[i][j] #get the int (edgeflow) representing the amount of observations going from node i to j
        group = random.sample(bin_dict[i], int(edgeflow)) #randomly sample x (edgeflow) unique elements from the list of observations in that category.
        q=j-k #q is the category index for a righhand side node
        for elem in group: #for each element in the randomly selected group list
          bin_dict[i].remove(elem) #remove the element from the list of observation in that category
          repair_data[elem] = index_dict[q] #Mutate repair data at the index of the observation (elem) with its new category (it was 0) which is the category index for the righthand side node it flows to
        if q in repair_bin_dict: #if the category index is already keyed
          repair_bin_dict[q].extend(group) #extend the list of observations with a new list of observations in that category
        else:
          repair_bin_dict[q] = group #otherwise key that category index and set it's value as the group list in that category
    new_feature = CategoricalFeature(repair_data) #initialize our new_feature (repaired feature)
    new_feature.bin_fulldata = repair_bin_dict
    return [new_feature,overflow]


def test():
  random.seed(10)
  test_feature = CategoricalFeature(["A","B","C","D","D","D","C","B","A","C","B","A"])
  desired_count_dict = {"A": 1, "B": 2, "C": 2, "D": 3}
  desired_category_count = lambda category : desired_count_dict[category]
  DG = test_feature.create_graph(desired_category_count)
  [new_feature, overflow] = test_feature.repair(DG)
  edges = [
  (0, 8, {'weight': 2}), (0, 4, {'weight': 0}), (0, 5, {'weight': 1}), (0, 6, {'weight': 1}), (0, 7, {'weight': 1}),
  (1, 8, {'weight': 2}), (1, 4, {'weight': 1}), (1, 5, {'weight': 0}), (1, 6, {'weight': 1}), (1, 7, {'weight': 1}),
  (2, 8, {'weight': 2}), (2, 4, {'weight': 1}), (2, 5, {'weight': 1}), (2, 6, {'weight': 0}), (2, 7, {'weight': 1}),
  (3, 8, {'weight': 2}), (3, 4, {'weight': 1}), (3, 5, {'weight': 1}), (3, 6, {'weight': 1}), (3, 7, {'weight': 0}),
  (4, 't', {'capacity': 1, 'weight': 0}), (5, 't', {'capacity': 2, 'weight': 0}), (6, 't', {'capacity': 2, 'weight': 0}),
  (7, 't', {'capacity': 3, 'weight': 0}),
  (8, 't', {'weight': 0}),
  ('s', 0, {'capacity': 3, 'weight': 0}), ('s', 1, {'capacity': 3, 'weight': 0}),
  ('s', 2, {'capacity': 3, 'weight': 0}), ('s', 3, {'capacity': 3, 'weight': 0})]
  new_data = [0, 0, 'C', 'D', 'D', 'D', 'C', 'B', 'A', 0, 'B', 0]
  print "CategoricalFeature has correct number of categories?", 4 == test_feature.num_bins
  print "Directed Graph has correct edges and edge weights?", DG.edges(data=True) == edges
  print "mincostFlow has correct overflow?", overflow == 4
  print "mincostFlow has correct output data?", new_feature.data == new_data

if __name__=="__main__": test()
