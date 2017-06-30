# NOTE:These settings and imports should be the only things that change
#       across experiments on different datasets
# TODO: Make this file generalizable to all datasets

from experiments.arrests.load_data import load_data
from repairers.CategoricRepairer import Repairer
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # Set the back-end
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"
if not os.path.exists(FIGURES_DIR):
  os.makedirs(FIGURES_DIR)

def run():
  feature_to_repair = 0
  repair_level = 1.0
  headers, train_data, test_data = load_data()
  orig_data = train_data
  repairer = Repairer(orig_data, feature_to_repair, repair_level)
  repaired_data = repairer.repair(test_data)

  features_to_graph = range(1, 13)
  for feature_to_graph in features_to_graph:
    header = headers[feature_to_graph]

    orig_groups = {}
    group_indices = {}
    for i, row in enumerate(orig_data):
      stratified_val = row[feature_to_repair]
      feature_val = row[feature_to_graph]
      if not stratified_val in orig_groups:
        orig_groups[stratified_val] = []
        group_indices[stratified_val] =[]
      orig_groups[stratified_val].append(feature_val)
      group_indices[stratified_val].append(i)
    rep_groups = {group:[repaired_data[i][feature_to_graph] for i in indices] for group, indices in group_indices.items()}

    data_dict = {0: {}, 1: {}}
    data_list = {0: {}, 1: {}}
    for group, data in orig_groups.items():
      data_dict[0][group] = {value: 0 for value in data}
      data_dict[1][group] = {value: 0 for value in data}
      for value in data:
        data_dict[0][group][value] += 1
      rep_data = rep_groups[group]
      for value in rep_data:
        if value in data_dict[1][group]:
          data_dict[1][group][value] += 1

    for group in orig_groups:
      data_list[0][group] = []
      data_list[1][group] = []
      for value in data_dict[0][group]:
        count0 = data_dict[0][group][value]
        count1 = data_dict[1][group][value]
        data_list[0][group].append(count0)
        data_list[1][group].append(count1)
      categories =  [value for value in data_dict[0][group]]
      n_categories = len(categories)

      count_group_orig = data_list[0][group]
      count_group_repaired = data_list[1][group]

      fig, ax = plt.subplots()

      index = np.arange(n_categories)
      bar_width = 0.35

      opacity = 0.4
      error_config = {'ecolor': '0.3'}

      rects1 = plt.bar(index, count_group_orig, bar_width,
                       alpha=opacity,
                       color='b',
                       label='Original')

      new_index = [i + bar_width for i in index]
      rects2 = plt.bar(new_index, count_group_repaired, bar_width,
                       alpha=opacity,
                       color='r',
                       label='Repaired')

      plt.xlabel('Categories')
      plt.ylabel('Count')
      plt.title(group + ' distribution over categories for feature ' + header)
      plt.xticks(index + bar_width, categories, rotation='vertical')
      plt.legend()

      plt.tight_layout()
      plt.savefig("figures/"+ header+ "_" + group + ".png")
      plt.savefig("{}/{}_{}.png".format(FIGURES_DIR, header, group))
      plt.clf()


if __name__=="__main__":
  run()

