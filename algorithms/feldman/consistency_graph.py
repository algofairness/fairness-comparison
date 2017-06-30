import csv
import sys

import matplotlib
matplotlib.use('Agg') # Set the back-end
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

def graph_prediction_consistency(directory, output_image_file):
  only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
  preds = ["{}/{}".format(directory, f) for f in only_files if ".predictions" in f]

  delim = ".audit.repaired_"

  ignored = ["original_train_data.predictions", "original_test_data.predictions"]

  file_groups = {}
  for pred in preds:
    if any(i in pred for i in ignored): continue
    feature = pred[len(directory)+1:pred.index(delim)] # Extract the feature name.
    if feature not in file_groups:
      file_groups[feature] = []
    file_groups[feature].append(pred)

  pred_groups = {}
  for feature, filenames in file_groups.items():
    pred_groups[feature] = []
    for filename in filenames:
      preds = load_pred_tups_from_predictions(filename)
      first_delim = filename.index(delim)+len(delim)
      second_delim = filename.index(".predictions")
      repair_level = float(filename[first_delim:second_delim])
      pred_groups[feature].append( (repair_level, preds) )
    pred_groups[feature].sort(key=lambda tup: tup[0]) # Sort by repair level.

  features = []
  y_axes = []
  for feature, pred_tups in pred_groups.items():
    orig = pred_tups[0][1]
    x_axis = [rep_level for rep_level, _ in pred_tups]
    y_axis = [similarity_to_original_preds(orig, tups) for _, tups in pred_tups]
    plt.plot(x_axis, y_axis, label=feature)

    features.append(feature)
    y_axes.append(y_axis)

  # Format and save the graph to an image file.
  plt.title("Similarity to Original Predictions")
  plt.axis([0,1,0,1.1]) # Make all the plots consistently sized.
  plt.xlabel("Repair Level")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig(output_image_file, bbox_inches='tight')
  plt.clf() # Clear the entire figure so future plots are empty.

  # Save the data used to generate that image file.
  with open(output_image_file + ".data", "w") as f:
    writer = csv.writer(f)
    headers = ["Repair Level"] + features
    writer.writerow(headers)
    for i, repair_level in enumerate(x_axis):
      writer.writerow([repair_level] + [y_vals[i] for y_vals in y_axes])


def load_pred_tups_from_predictions(filename):
  with open(filename) as f:
    reader = csv.reader(f)
    reader.next # Skip the headers.
    return [(r,p) for _,r,p in reader]

def similarity_to_original_preds(orig_pred_tups, new_pred_tups):
  matches = 0.0
  total = 0.0
  for orig, new in zip(orig_pred_tups, new_pred_tups):
    _,orig_pred = orig
    _,new_pred = new
    if orig_pred == new_pred:
      matches += 1
    total += 1

  return matches/total


if __name__=="__main__":
  if len(sys.argv) >= 2:
    directory = sys.argv[1]
    output_image = directory + "/similarity_to_original_predictions.png"
    graph_prediction_consistency(directory, output_image)
    print "Written to: {}".format(output_image)
