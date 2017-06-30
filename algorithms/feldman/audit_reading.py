import json
import os
import random
import csv

import matplotlib
matplotlib.use('Agg') # Set the back-end
import matplotlib.pyplot as plt


def load_audit_confusion_matrices(filename):
  """
  Loads a confusion matrix in a two-level dictionary format.

  For example, the confusion matrix of a 75%-accurate model
  that predicted 15 values (and mis-classified 5) may look like:
  {"A": {"A":10, "B": 5}, "B": {"B":5}}

  Note that raw boolean values are translated into strings, such that
  a value that was the boolean True will be returned as the string "True".
  """

  with open(filename) as audit_file:
    audit_file.next() # Skip the first line.

    # Extract the confusion matrices and repair levels from the audit file.
    confusion_matrices = []
    for line in audit_file:
      separator = ":"
      separator_index = line.index(separator)

      repair_level = float(line[:separator_index])
      raw_confusion_matrix = line[separator_index + len(separator):-1]
      confusion_matrix = json.loads( raw_confusion_matrix.replace("'","\"") )
      confusion_matrices.append( (repair_level, confusion_matrix) )

  # Sort the repair levels in case they are out of order for whatever reason.
  confusion_matrices.sort(key = lambda pair: pair[0])
  return confusion_matrices


def graph_audit(filename, measurers, output_image_file):
  with open(filename) as audit_file:
    header_line = audit_file.readline()[:-1] # Remove the trailing endline.

  confusion_matrices = load_audit_confusion_matrices(filename)

  x_axis = [repair_level for repair_level, _ in confusion_matrices]
  y_axes = []

  # Graph the results for each requested measurement.
  for measurer in measurers:
    y_axis = [measurer(matrix) for _, matrix in confusion_matrices]
    plt.plot(x_axis, y_axis, label=measurer.__name__)
    y_axes.append(y_axis)

  # Format and save the graph to an image file.
  plt.title(header_line)
  plt.axis([0,1,0,1.1]) # Make all the plots consistently sized.
  plt.xlabel("Repair Level")
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig(output_image_file, bbox_inches='tight')
  plt.clf() # Clear the entire figure so future plots are empty.

  # Save the data used to generate that image file.
  with open(output_image_file + ".data", "w") as f:
    writer = csv.writer(f)
    headers = ["Repair Level"] + [calc.__name__ for calc in measurers]
    writer.writerow(headers)
    for i, repair_level in enumerate(x_axis):
      writer.writerow([repair_level] + [y_vals[i] for y_vals in y_axes])

def graph_audits(filenames, measurer, output_image_file):
  features = []
  y_axes = []
  for filename in filenames:
    with open(filename) as audit_file:
      header_line = audit_file.readline()[:-1] # Remove the trailing endline.
      feature = header_line[header_line.index(":")+1:]

    confusion_matrices = load_audit_confusion_matrices(filename)
    x_axis = [repair_level for repair_level, _ in confusion_matrices]
    y_axis = [measurer(matrix) for _, matrix in confusion_matrices]
    plt.plot(x_axis, y_axis, label=feature)

    features.append(feature)
    y_axes.append(y_axis)

  # Format and save the graph to an image file.
  plt.title(measurer.__name__)
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


def rank_audit_files(filenames, measurer):
  scores = []
  for filename in filenames:
    with open(filename) as audit_file:
      header_line = audit_file.readline()[:-1] # Remove the trailing endline.
      feature = header_line[header_line.index(":")+1:]

    confusion_matrices = load_audit_confusion_matrices(filename)
    _, start_matrix = confusion_matrices[0]
    _, end_matrix = confusion_matrices[-1]
    score_difference = measurer(start_matrix)-measurer(end_matrix)
    scores.append( (feature, score_difference) )

  scores.sort(key = lambda score_tup: score_tup[1], reverse=True)
  return scores


def group_audit_ranks(filenames, measurer, similarity_bound=0.05):
  """
  Given a list of audit files, rank them using the `measurer` and
  return the features that never deviate more than `similarity_bound`
  across repairs.
  """

  def _partition_groups(feature_scores):
    groups = []
    for feature, score in feature_scores:
      added_to_group = False

      # Check to see if the feature belongs in a group with any other features.
      for i, group in enumerate(groups):
        mean_score, group_feature_scores = group
        if abs(mean_score - score) < similarity_bound:
          groups[i][1].append( (feature, score) )

          # Recalculate the representative mean.
          groups[i][0] = sum([s for _, s in group_feature_scores])/len(group_feature_scores)
          added_to_group = True
          break

      # If this feature did not much with the current groups, create another group.
      if not added_to_group:
        groups.append( [score, [(feature,score)]] )

    # Return just the features.
    return [[feature for feature, score in group] for _, group in groups]


  score_dict = {}
  features = []
  for filename in filenames:
    with open(filename) as audit_file:
      header_line = audit_file.readline()[:-1] # Remove the trailing endline.
      feature = header_line[header_line.index(":")+1:]
      features.append(feature)

    confusion_matrices = load_audit_confusion_matrices(filename)
    for rep_level, matrix in confusion_matrices:
      score = measurer(matrix)
      if rep_level not in score_dict:
        score_dict[rep_level] = {}
      score_dict[rep_level][feature] = score

  # Sort by repair level increasing repair level.
  score_keys = sorted(score_dict.keys())

  groups = [features]
  while score_keys:
    key = score_keys.pop()
    new_groups = []
    for group in groups:
      group_features = [(f, score_dict[key][f]) for f in group]
      sub_groups = _partition_groups(group_features)
      new_groups.extend(sub_groups)
    groups = new_groups

  return groups


def test():
  TMP_DIR = "tmp"
  if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

  test_contents = "GFA Audit for: Test Feature\n0.0:{'A': {'B': 100}, 'B': {'B': 199}}\n0.1:{'A': {'B': 100}, 'B': {'B': 199}}\n0.5:{'A': {'B': 100}, 'B': {'B': 199}}\n1.0:{'A': {'B': 100}, 'B': {'B': 199}}\n"
  test_filenames = [TMP_DIR + "/test_audit_1.audit",
                    TMP_DIR + "/test_audit_2.audit"]

  # Prepare the sample audit files.
  for filename in test_filenames:
    with open(filename, "w") as f:
      f.write(test_contents)

  # A mock measurement measurer that returns a random number.
  def mock_measurer(conf_matrix):
    return random.random()

  # Perform the audit and save it an output image.
  measurers = [mock_measurer, mock_measurer]
  output_image = TMP_DIR + "/test_image.png"
  graph_audit(test_filenames[0], measurers, output_image) # Only need to test 1.

  file_not_empty = os.path.getsize(output_image) > 0
  print "image file generated? --", file_not_empty

  file_not_empty = os.path.getsize(output_image + ".data") > 0
  print "data file generated? --", file_not_empty

  ranked_features = rank_audit_files(test_filenames, mock_measurer)
  print "ranked features sorted? --", ranked_features[0] > ranked_features[1]

  output_image = TMP_DIR + "/test_image2.png"
  graph_audits(test_filenames, mock_measurer, output_image)
  file_not_empty = os.path.getsize(output_image) > 0
  print "ranked image file generated? --", file_not_empty


if __name__=="__main__":
  test()
