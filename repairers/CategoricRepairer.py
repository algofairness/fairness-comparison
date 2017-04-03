from itertools import product
from collections import defaultdict

from AbstractRepairer import AbstractRepairer
from CategoricalFeature import CategoricalFeature
from calculators import get_median
from SparseList import SparseList

import random
import math
from copy import deepcopy


class Repairer(AbstractRepairer):
  def repair(self, data_to_repair):
    num_cols = len(data_to_repair[0])
    col_ids = range(num_cols)

    # Get column type information
    col_types = ["Y"]*len(col_ids)
    for i, col in enumerate(col_ids):
      if i in self.features_to_ignore:
        col_types[i] = "I"
      elif i == self.feature_to_repair:
        col_types[i] = "X"

    col_type_dict = {col_id: col_type for col_id, col_type in zip(col_ids, col_types)}

    not_I_col_ids = filter(lambda x: col_type_dict[x] != "I", col_ids)
    cols_to_repair = filter(lambda x: col_type_dict[x] in "YX", col_ids)

    # To prevent potential perils with user-provided column names, map them to safe column names
    safe_stratify_cols = [self.feature_to_repair]

    # Extract column values for each attribute in data
    # Begin by initializing keys and values in dictionary
    data_dict = {col_id: [] for col_id in col_ids}

    # Populate each attribute with its column values
    for row in data_to_repair:
      for i in col_ids:
        data_dict[i].append(row[i])


    repair_types = {}
    for col_id, values in data_dict.items():
      if all(isinstance(value, float) for value in values):
        repair_types[col_id] = float
      elif all(isinstance(value, int) for value in values):
        repair_types[col_id] = int
      else:
        repair_types[col_id] = str

    """
     Create unique value structures: When performing repairs, we choose median values. If repair is partial, then values will be modified to some intermediate value between the original and the median value. However, the partially repaired value will only be chosen out of values that exist in the data set.  This prevents choosing values that might not make any sense in the data's context.  To do this, for each column, we need to sort all unique values and create two data structures: a list of values, and a dict mapping values to their positions in that list. Example: There are unique_col_vals[col] = [1, 2, 5, 7, 10, 14, 20] in the column. A value 2 must be repaired to 14, but the user requests that data only be repaired by 50%. We do this by finding the value at the right index:
       index_lookup[col][2] = 1
       index_lookup[col][14] = 5
       this tells us that unique_col_vals[col][3] = 7 is 50% of the way from 2 to 14.
    """
    unique_col_vals = {}
    index_lookup = {}
    for col_id in not_I_col_ids:
      col_values = data_dict[col_id]
      # extract unique values from column and sort
      col_values = sorted(list(set(col_values)))
      unique_col_vals[col_id] = col_values
      # look up a value, get its position
      index_lookup[col_id] = {col_values[i]: i for i in range(len(col_values))}

    """
     Make a list of unique values per each stratified column.  Then make a list of combinations of stratified groups. Example: race and gender cols are stratified: [(white, female), (white, male), (black, female), (black, male)] The combinations are tuples because they can be hashed and used as dictionary keys.  From these, find the sizes of these groups.
    """
    unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
    all_stratified_groups = list(product(*unique_stratify_values))
    # look up a stratified group, and get a list of indices corresponding to that group in the data
    stratified_group_indices = defaultdict(list)

    # Find the number of unique values for each strat-group, organized per column.
    val_sets = {group: {col_id:set() for col_id in cols_to_repair}
                                     for group in all_stratified_groups}
    for i, row in enumerate(data_to_repair):
      group = tuple(row[col] for col in safe_stratify_cols)
      for col_id in cols_to_repair:
        val_sets[group][col_id].add(row[col_id])

      # Also remember that this row pertains to this strat-group.
      stratified_group_indices[group].append(i)


    """
     Separate data by stratified group to perform repair on each Y column's values given that their corresponding protected attribute is a particular stratified group. We need to keep track of each Y column's values corresponding to each particular stratified group, as well as each value's index, so that when we repair the data, we can modify the correct value in the original data. Example: Supposing there is a Y column, "Score1", in which the 3rd and 5th scores, 70 and 90 respectively, belonged to black women, the data structure would look like: {("Black", "Woman"): {Score1: [(70,2),(90,4)]}}
    """
    stratified_group_data = {group: {} for group in all_stratified_groups}
    for group in all_stratified_groups:
      for col_id, col_dict in data_dict.items():
        # Get the indices at which each value occurs.
        indices = {}
        for i in stratified_group_indices[group]:
          value = col_dict[i]
          if value not in indices:
            indices[value] = []
          indices[value].append(i)

        stratified_col_values = [(occurs, val) for val, occurs in indices.items()]
        stratified_col_values.sort(key=lambda tup: tup[1])
        stratified_group_data[group][col_id] = stratified_col_values

    mode_feature_to_repair = get_mode(data_dict[self.feature_to_repair])

    # Repair Data and retrieve the results
    for col_id in cols_to_repair:
      # which bucket value we're repairing
      group_offsets = {group: 0 for group in all_stratified_groups}
      col = data_dict[col_id]

      num_quantiles = min(len(val_sets[group][col_id]) for group in all_stratified_groups)
      quantile_unit = 1.0/num_quantiles

      if repair_types[col_id] in {int, float}:
        for quantile in range(num_quantiles):
          median_at_quantiles = []
          indices_per_group = {}

          for group in all_stratified_groups:
            group_data_at_col = stratified_group_data[group][col_id]
            num_vals = len(group_data_at_col)
            offset = int(round(group_offsets[group]*num_vals))
            number_to_get = int(round((group_offsets[group] + quantile_unit)*num_vals) - offset)
            group_offsets[group] += quantile_unit

            if number_to_get > 0:

              # Get data at this quantile from this Y column such that stratified X = group
              offset_data = group_data_at_col[offset:offset+number_to_get]
              indices_per_group[group] = [i for val_indices, _ in offset_data for i in val_indices]
              values = sorted([float(val) for _, val in offset_data])

              # Find this group's median value at this quantile
              median_at_quantiles.append( get_median(values) )

          # Find the median value of all groups at this quantile (chosen from each group's medians)
          median = get_median(median_at_quantiles)
          median_val_pos = index_lookup[col_id][median]

          # Update values to repair the dataset.
          for group in all_stratified_groups:
            for index in indices_per_group[group]:
              original_value = col[index]

              current_val_pos = index_lookup[col_id][original_value]
              distance = median_val_pos - current_val_pos # distance between indices
              distance_to_repair = int(round(distance * self.repair_level))
              index_of_repair_value = current_val_pos + distance_to_repair
              repaired_value = unique_col_vals[col_id][index_of_repair_value]

              # Update data to repaired valued
              data_dict[col_id][index] = repaired_value

      #Categorical Repair is done below
      elif repair_types[col_id] in {str}:
        feature = CategoricalFeature(col)
        categories = feature.bin_index_dict.keys()

        group_features = get_group_data(all_stratified_groups, stratified_group_data, col_id)

        categories_count = get_categories_count(categories, all_stratified_groups, group_features)

        categories_count_norm = get_categories_count_norm(categories, all_stratified_groups, categories_count, group_features)

        median = get_median_per_category(categories, categories_count_norm)

        # Partially fill-out the generator functions to simplify later calls.
        dist_generator = lambda group_index, category : gen_desired_dist(group_index, category, col_id, median, self.repair_level, categories_count_norm, self.feature_to_repair, mode_feature_to_repair)

        count_generator = lambda group_index, group, category : gen_desired_count(group_index, group, category, median, group_features, self.repair_level, categories_count)

        group_features, overflow = flow_on_group_features(all_stratified_groups, group_features, count_generator)

        group_features, assigned_overflow, distribution = assign_overflow(all_stratified_groups, categories, overflow, group_features, dist_generator)

        # Return our repaired feature in the form of our original dataset
        for group in all_stratified_groups:
          indices = stratified_group_indices[group]
          for i, index in enumerate(indices):
            repaired_value = group_features[group].data[i]
            data_dict[col_id][index] = repaired_value

    # Replace stratified groups with their mode value, to remove it's information
    repaired_data = []
    for i, orig_row in enumerate(data_to_repair):
      new_row = [orig_row[j] if j not in cols_to_repair else data_dict[j][i] for j in col_ids]
      repaired_data.append(new_row)

    return repaired_data

def get_group_data(all_stratified_groups,stratified_group_data, col_id):
  group_features={}
  for group in all_stratified_groups:
    points = [(i, val) for indices, val in stratified_group_data[group][col_id] for i in indices]
    points = sorted(points, key=lambda x: x[0]) # Sort by index
    values = [value for _, value in points]

    # send values to CategoricalFeature object, which bins the data into categories
    group_features[group] = CategoricalFeature(values)
  return group_features


# Count the observations in each category. e.g. categories_count[1] = {'A':[1,2,3], 'B':[3,1,4]}, for column 1, category 'A' has 1 observation from group 'x', 2 from 'y', ect.
def get_categories_count(categories, all_stratified_groups, group_feature):
  # Grossness for speed efficiency. Don't worry, it make me sad, too.
  count_dict={cat: SparseList(data=(group_feature[group].category_count[cat] if cat in group_feature[group].category_count else 0 for group in all_stratified_groups)) for cat in categories}

  return count_dict

# Find the normalized count for each category, where normalized count is count divided by the number of people in that group
def get_categories_count_norm(categories, all_stratified_groups, count_dict, group_features):
  # Forgive me, Father, for I have sinned in bringing this monstrosity into being.
  norm = {cat: SparseList(data=(count_dict[cat][i] * (1.0/len(group_features[group].data)) if group_features[group].data else 0.0 for i,group in enumerate(all_stratified_groups))) for cat in categories}
  return norm

# Find the median normalized count for each category
def get_median_per_category(categories, categories_count_norm):
  return {cat: get_median(categories_count_norm[cat]) for cat in categories}

# Generate the desired distribution and desired "count" for a given group-category-feature combination.
def gen_desired_dist(group_index, cat, col_id, median, repair_level, norm_counts, feature_to_remove, mode):
      if feature_to_remove == col_id:
        return 1 if cat==mode else (1-repair_level)*norm_counts[cat][group_index]
      else:
        return (1 - repair_level)*norm_counts[cat][group_index] + (repair_level*median[cat])

def gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count):
      med=median[category]
      size = len(group_features[group].data)
      # des-proportion = (1-lambda)*original-count  + (lambda)*median-count
      count = categories_count[category][group_index]
      des_count = math.floor(((1-repair_level)*count)+(repair_level)*med*size)
      return des_count

 # Run Max-flow to distribute as many observations to categories as possible. Overflow are those observations that are left over
def flow_on_group_features(all_stratified_groups, group_features, repair_generator):
  dict1= {}
  dict2={}
  for i, group in enumerate(all_stratified_groups):
    feature = group_features[group]
    count_generator = lambda category : repair_generator(i, group, category)

    # Create directed graph from nodes that supply the original countes to nodes that demand the desired counts, with a overflow node as total desired count is at most total original counts
    DG=feature.create_graph(count_generator)

    # Run max-flow, and record overflow count (total and per-group)
    new_feature,overflow = feature.repair(DG)
    dict2[group] = overflow

    # Update our original values with the values from max-flow, Note: still missing overflowed observations
    dict1[group] = new_feature

  return dict1, dict2

# Assign overflow observations to categories based on the group's desired distribution
def assign_overflow(all_stratified_groups, categories, overflow, group_features, repair_generator):
  feature = deepcopy(group_features)
  assigned_overflow = {}
  desired_dict_list = {}
  for group_index, group in enumerate(all_stratified_groups):
    # Calculate the category proportions.
    dist_generator = lambda cat: repair_generator(group_index, cat)
    cat_props = map(dist_generator,categories)

    if all(elem==0 for elem in cat_props): #TODO: Check that this is correct!
      cat_props = [1.0/len(cat_props)] * len(cat_props)
    s = float(sum(cat_props))
    cat_props = [elem/s for elem in cat_props]
    desired_dict_list[group] = cat_props
    assigned_overflow[group] = {}
    for i in range(int(overflow[group])):
      distribution_list = desired_dict_list[group]
      number = random.uniform(0, 1)
      cat_index = 0
      tally = 0
      for j in range(len(distribution_list)):
        value=distribution_list[j]
        if number < (tally+value):
          cat_index = j
          break
        tally += value
      assigned_overflow[group][i] = categories[cat_index]
    # Actually do the assignment
    count = 0
    for i, value in enumerate(group_features[group].data):
      if value == 0:
        (feature[group].data)[i] = assigned_overflow[group][count]
        count += 1
  return feature, assigned_overflow, desired_dict_list

def get_mode(values):
  counts = {}
  for value in values:
    counts[value] = 1 if value not in counts else counts[value]+1
  mode_tuple = max(counts.items(), key=lambda tup: tup[1])
  return mode_tuple[0]


def test():
  test_minimal()
  test_get_group_data()
  test_get_categories_count()
  test_get_categories_count_norm()
  test_get_median_per_category()
  test_gen_desired_count()
  test_gen_desired_dist()
  test_assign_overflow()
  test_categorical()
  test_repeated_values()

def test_repeated_values():
  all_data = [
  ["x","A"], ["x","B"], ["x","C"], ["x","D"], ["x","E"],
  ["y","F"], ["y","G"], ["y","H"], ["y","I"],
  ["z","J"], ["z","K"], ["z","L"], ["z","M"], ["z","N"], ["z","O"]]

  random.seed(10)

  repair_level=.5
  feature_to_repair = 0
  repairer = Repairer(all_data, feature_to_repair, repair_level)
  repaired_data=repairer.repair(all_data)

  correct_repaired_data = [
  ['x', 'E'], ['x', 'C'], ['x', 'C'], ['z', 'E'], ['x', 'B'], 
  ['y', 'I'], ['y', 'I'], ['y', 'G'], ['z', 'H'], 
  ['z', 'N'], ['z', 'N'], ['z', 'L'], ['z', 'K'], ['z', 'K'], ['z', 'K']]

  print "Test unique values -- .5 repaired_data altered?", repaired_data != all_data
  print "Test unique values -- .5 repaired_data correct?", repaired_data == correct_repaired_data

  all_data = [
  ["x","A"], ["x","A"], ["x","A"], ["x","A"], ["x","A"],
  ["y","A"], ["y","A"], ["y","A"], ["y","A"],
  ["z","A"], ["z","A"], ["z","A"], ["z","A"], ["z","A"], ["z","A"]]

  random.seed(10)

  repair_level=.5
  feature_to_repair = 0
  repairer = Repairer(all_data, feature_to_repair, repair_level)
  repaired_data=repairer.repair(all_data)

  correct_repaired_data = [
  ['x', 'A'], ['x', 'A'], ['x', 'A'], ['z', 'A'], ['x', 'A'], 
  ['y', 'A'], ['y', 'A'], ['y', 'A'], ['z', 'A'], 
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A']]


  print "Test repeated values -- .5 repaired_data altered?", repaired_data != all_data
  print "Test repeated values -- .5 repaired_data correct?", repaired_data == correct_repaired_data

def test_minimal():
  class_1 = [[float(i),"A"] for i in xrange(0, 100)]
  class_2 = [[float(i),"B"] for i in xrange(101, 200)] # Thus, "A" is mode class.
  data = class_1 + class_2
  print "HERE"

  feature_to_repair = 1
  repairer = Repairer(data, feature_to_repair, 1)
  repaired_data = repairer.repair(data)
  print "Minimal Dataset -- repaired_data altered?", repaired_data != data

  mode = get_mode([row[feature_to_repair] for row in data])
  print "Minimal Dataset -- mode is true mode?", mode=="A"
  print "Minimal Dataset -- mode value as feature_to_repair?", all(row[feature_to_repair] == mode for row in repaired_data)

def test_get_group_data():
  group_features = {}
  group_size = {}
  col_id = 1
  all_stratified_groups = [('y',),('z',)]
  stratified_group_data = {('y',): {1: [([4, 7, 5], 'A'),([3, 2, 6], 'B'), ([], 'C')]},\
                           ('z',): {1: [([9, 1, 10], 'A'), ([], 'B'), ([11], 'C')]}}
  group_features[col_id] = get_group_data(all_stratified_groups, stratified_group_data, col_id)

  print "Test get_group_data -- group features correct?", \
    [group_features[col_id][group].data for group in all_stratified_groups] == [['B','B','A','A','B','A'],['A', 'A', 'A', 'C']]
  #print "Test get_group_data -- group sizes correct?", group_size[col_id] == {('y',): 6, ('z',):4}

def test_get_categories_count():
  categories_count = {}
  categories = {1:['A','B','C','D']}
  all_stratified_groups = [('y',),('z',)]
  col_id = 1
  group_features = {1:{('y',): CategoricalFeature(['C','A','C','B','A','C']),\
                      ('z',): CategoricalFeature(['B','B','D','D'])}}\

  categories_count[col_id] = get_categories_count(categories[col_id], all_stratified_groups, group_features[col_id])
  #SparseList makes us test for correctness strangely
  print "Test get_categories_count -- category counts correct?",\
   categories_count[col_id]['A'][0] ==2 and categories_count[col_id]['C'][0] ==3 and categories_count[col_id]['B'][0] ==1\
   and categories_count[col_id]['B'][1] ==2 and categories_count[col_id]['D'][1] ==2

def test_get_categories_count_norm():
  categories_count_norm = {}
  categories = {1:['A','B']}
  all_stratified_groups = [('y',),('z',)]
  col_id = 1
  categories_count = {1: {'A':[4,0],'B':[16,0]}}
 
  #group_size = {1: {('y',): 20,('z',): 0}}
  group_features = {1:{('y',): CategoricalFeature(['A','A','A','A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
                        ('z',): CategoricalFeature([])}}
  categories_count_norm[col_id] = get_categories_count_norm(categories[col_id], all_stratified_groups, categories_count[col_id], group_features[col_id])
  #SparseList makes us test for correctness strangely
  print "Test get_categories_count_norm -- normalized category counts correct?",\
    categories_count_norm[col_id]['A'][0] == 0.2 and  categories_count_norm[col_id]['B'][0] == 0.8

def test_get_median_per_category():
  categories = {1:['A','B','C','D']}
  col_id =1
  categories_count_norm = {1:{'A':[0.25,0.0],'C':[0.3,0.0],'B':[0.4,0.0],  'D':[0.6,0.4]}}
  median = get_median_per_category(categories[col_id], categories_count_norm[col_id])
  print "Test get_median_per_category -- medians are correct?", median == {'A':0.0,'C':0.0,'B':0.0,  'D':0.4}

def test_gen_desired_count():
  #Case 1: feature with category with no values
  group_index = 0
  group = ('y',)
  median = {'A': 0.0 ,'B':0.0}
  group_features =  {('y',): CategoricalFeature(['A','A','A','A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
                        ('z',): CategoricalFeature([])}
  repair_level = .25
  categories_count = {'A':[4,0],'B':[16,0]}
  category = 'B'
  des_countB = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  category = 'A'
  des_countA = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  print "Test gen_desired_count -- desired count correct for feature with category with no values?", des_countB==12 and des_countA==3

  #Case 2: feature with regular categories
  group_index = 0
  group = ('y',)
  median = {'A': 0.2 ,'B':0.75}
  group_features =  {('y',): CategoricalFeature(['A','A','A','A', 'A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
                        ('z',): CategoricalFeature(['A','A','B','B','B','B','B','B','B','B'])}
  repair_level = 0.001
  categories_count = {'A':[5,2],'B':[15,8]}
  category = 'B'
  des_countB = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  category = 'A'
  des_countA = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  print "Test gen_desired_count -- desired count correct for standard feature?", des_countB==15 and des_countA==4 

  #Case 3: Repair feature after having repaired the other features on such feature 
  col_id = 0
  median = {'Y': 0.0 ,'Z':0.0}
  group_features =  {('y',): CategoricalFeature(['Y','Y','Y','Y']),
                       ('z',): CategoricalFeature(['Z','Z'])}
  repair_level = .5
  categories_count = {'Y':[4,0],'Z':[0,2]}
  mode_feature = 'Y' 
  group_index = 0
  group = ('y',)
  category = 'Y'
  des_countY = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  group_index = 1
  group = ('z',)
  category = 'Z'
  des_countZ = gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)
  print "Test gen_desired_count -- desired count correct for mode category when repairing feature to remove?", des_countY==2 and des_countZ==1
  # If you are confused why desired count for Z is 2, it is beacuse our group_index is for group y 
  
def test_gen_desired_dist():
  #Case 1: feature with category with no values
  group_index = 0
  #group = ('y',)
  col_id = 1
  median = {'A': 0.0 ,'B':0.0}
  #group_features =  {('y',): CategoricalFeature(['A','A','A','A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
  #                      ('z',): CategoricalFeature([])}
  repair_level = .5
  categories_count_norm =  {'A':[0.2,0.0],'B':[0.8,0.0]}
  #categories_count = {'A':[4,0],'B':[16,0]}
  feature_to_remove = 0
  mode_feature = 'B' 
  category = 'B'
  des_distB = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  category = 'A'
  des_distA = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  print "Test gen_desired_dist -- desired distribution correct for feature with category with no values?", des_distB == .4 and des_distA == .1
  
  #Case 2: feature with regular categories
  group_index = 0
  #group = ('y',)
  median = {'A': 0.2 ,'B':0.75}
  #group_features =  {('y',): CategoricalFeature(['A','A','A','A', 'A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
  #                      ('z',): CategoricalFeature(['A','A','B','B','B','B','B','B','B','B'])}
  repair_level = 0.001
  categories_count_norm =  {'A':[0.25,0.2],'B':[0.75,0.8]}
  #categories_count = {'A':[5,2],'B':[15,8]}
  feature_to_remove = 0
  category = 'B'
  des_distB = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  category = 'A'
  des_distA = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  print "Test gen_desired_dist -- desired distribution correct for standard feature?", des_distB == .75 and des_distA == .24995 

  #Case 3: Repair feature after having repaired the other features on such feature 
  col_id = 0
  median = {'Y': 0.0 ,'Z':0.0}
  #group_features =  {('y',): CategoricalFeature(['Y','Y','Y','Y']),
  #                     ('z',): CategoricalFeature(['Z','Z'])}
  repair_level = .5
  categories_count_norm =  {'Y':[1.0,0.0],'Z':[0.0,1.0]}
  #categories_count = {'Y':[4,0],'Z':[0,2]}
  feature_to_remove = 0
  mode_feature = 'Y' 
  group_index = 0
  category = 'Y'
  des_distY = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  group_index = 1
  category = 'Z'
  des_distZ = gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  print "Test gen_desired_dist -- desired distribution correct for mode category when repairing feature to remove?", des_distY==1 and des_distZ==.5

def test_assign_overflow():
  group_index = 0
  group = ('y',)
  category = 'B'
  col_id = 1
  median = {'A': 0.0 ,'B':0.0}
  group_features =  {('y',): CategoricalFeature(['A','A','A','A','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B']),
                      ('z',): CategoricalFeature([])}
  repair_level = 1
  categories_count_norm =  {'A':[0.2,0.0],'B':[0.8,0.0]}
  categories_count = {'A':[4,0],'B':[16,0]}
   
  feature_to_remove = 0
  mode_feature = 'B' 
  dist_generator = lambda group_index, category : gen_desired_dist(group_index, category, col_id, median, repair_level, categories_count_norm, feature_to_remove, mode_feature)
  count_generator = lambda group_index, group, category : gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count)

  random.seed(10)
  
  all_stratified_groups = [('y',),('z',)]
  categories = ['A','B']
  col_id = 1
  overflow = {('y',):2,('z',):2}
  group_features = {('y',):CategoricalFeature(['A','A','B','B',0,0]), ('z',): CategoricalFeature(['B',0,0])}

  feature, assigned_overflow, desired_dict_list = assign_overflow(all_stratified_groups, categories, overflow, group_features, dist_generator)
  print "Test assign_overflow -- updated group features correct?", \
   [feature[group].data for group in all_stratified_groups] ==[['A', 'A', 'B', 'B', 'B', 'A'], ['B', 'B', 'A']]
  print "Test assign_overflow -- assigned overflow correctly?", assigned_overflow == {('y',): {0: 'B', 1: 'A'}, ('z',): {0: 'B', 1: 'A'}}
  print "Test assign_overflow -- distribution correct?", desired_dict_list == {('y',): [0.5, 0.5], ('z',): [0.5, 0.5]}

def test_categorical():
  all_data = [
  ["x","A"], ["x","A"], ["x","B"], ["x","B"], ["x","B"],
  ["y","A"], ["y","A"], ["y","A"], ["y","B"],
  ["z","A"], ["z","A"], ["z","A"], ["z","A"], ["z","A"], ["z","B"]]

  random.seed(10)

  repair_level=1
  feature_to_repair = 0
  repairer = Repairer(all_data, feature_to_repair, repair_level)
  repaired_data=repairer.repair(all_data)

  correct_repaired_data = [
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'B'], 
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'B'], 
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'B']]

  print "Categorical Minimal Dataset -- full repaired_data altered?", repaired_data != all_data
  print "Categorical Minimal Dataset -- full repaired_data correct?", repaired_data == correct_repaired_data

  repair_level=0.5
  feature_to_repair = 0
  repairer = Repairer(all_data, feature_to_repair, repair_level)
  part_repaired_data=repairer.repair(all_data)

  correct_part_repaired_data = [
  ['x', 'A'], ['x', 'A'], ['z', 'B'], ['z', 'B'], ['x', 'B'], 
  ['y', 'A'], ['y', 'A'], ['y', 'A'], ['y', 'B'], 
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'B']]

  print "Categorical Minimal Dataset -- partial (.5) repaired_data altered?", part_repaired_data != all_data
  print "Categorical Minimal Dataset -- partial (.5) repaired_data correct?", part_repaired_data == correct_part_repaired_data

  repair_level=0.2
  feature_to_repair = 0
  repairer = Repairer(all_data, feature_to_repair, repair_level)
  part_repaired_data=repairer.repair(all_data)

  correct_part2_repaired_data =  [
  ['x', 'A'], ['x', 'A'], ['x', 'B'], ['x', 'B'], ['z', 'B'], 
  ['y', 'A'], ['y', 'A'], ['y', 'A'], ['y', 'B'], 
  ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'A'], ['z', 'B']]

  print "Categorical Minimal Dataset -- partial (.2) repaired_data altered?", part_repaired_data != all_data
  print "Categorical Minimal Dataset -- partial (.2) repaired_data correct?", part_repaired_data == correct_part2_repaired_data


if __name__== "__main__":
  test()
