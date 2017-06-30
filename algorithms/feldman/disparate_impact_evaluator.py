# NOTE: These settings and imports should be the only things that change
#       across experiments on different datasets and ML model types.

from disparate_impact import disparate_impact
from consistency_graph import *


from os import listdir
from os.path import isfile, join

def load_trip_from_predictions(filename):
  with open(filename) as f:
    reader = csv.reader(f)
    reader.next # Skip the headers.
    return [(f,r,p) for f,r,p in reader]

def graph_disparate_impact_accuracy(directory, output_image_file):
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
      preds = load_trip_from_predictions(filename)
      first_delim = filename.index(delim)+len(delim)
      second_delim = filename.index(".predictions")
      repair_level = float(filename[first_delim:second_delim])
      pred_groups[feature].append( (repair_level, preds) )
    pred_groups[feature].sort(key=lambda tup: tup[0]) # Sort by repair level.

  protected_groups = ["WHITE","BLACK", "UNKNOWN", "ASIAN/PACIFIC ISLANDER", "AMERICAN INDIAN/ALEUTIAN"]
  unprotected_group = "WHITE"
  features = []
  y_axes = []
  for feature, pred_tups in pred_groups.items():
    if feature == "RACE":
      for protected_group in protected_groups:
        #TODO clean this bit of code up, it's hacky right now
        real_pred_tups=[]
        for repair_level, triples in pred_tups:
          tups = []
          for triple in triples:
            tups.append((triple[1],triple[2]))
          real_pred_tups.append((repair_level,tups))
        
        orig = []
        for triple in pred_tups[0][1]:
          orig.append((triple[1],triple[2]))
        
        x_axis = [disparate_impact(triples[1:], unprotected_group, protected_group) for _,triples in pred_tups] 
        y_axis = [similarity_to_original_preds(orig, tups) for _, tups in real_pred_tups]
        plt.plot(x_axis, y_axis, label=protected_group)
        features.append(protected_group)
        y_axes.append(y_axis)

      # Format and save the graph to an image file.
      plt.title("Similarity to Original Predictions")
      plt.axis([.4,1.4,.7,1.1]) # Make all the plots consistently sized.
      plt.xlabel("Disparate Impact: Pr(Good Outcome given Race = X) / Pr(Good Outcome given Race = WHITE)")
      plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
      plt.savefig(output_image_file, bbox_inches='tight')
      plt.clf() # Clear the entire figure so future plots are empty.

      # Save the data used to generate that image file.
      with open(output_image_file + ".data", "w") as f:
        writer = csv.writer(f)
        headers = ["Disparate Impact"] + features
        writer.writerow(headers)
        for i, repair_level in enumerate(x_axis):
          writer.writerow([repair_level] + [y_vals[i] for y_vals in y_axes])

def graph_repair_level_disparate_impact(directory, output_image_file):
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
      preds = load_trip_from_predictions(filename)
      first_delim = filename.index(delim)+len(delim)
      second_delim = filename.index(".predictions")
      repair_level = float(filename[first_delim:second_delim])
      pred_groups[feature].append( (repair_level, preds) )
    pred_groups[feature].sort(key=lambda tup: tup[0]) # Sort by repair level.

  protected_groups = ["WHITE","BLACK", "UNKNOWN", "ASIAN/PACIFIC ISLANDER", "AMERICAN INDIAN/ALEUTIAN"]
  unprotected_group = "WHITE"
  features = []
  y_axes = []
  for feature, pred_tups in pred_groups.items():
    if feature == "RACE":
      for protected_group in protected_groups:
        #TODO Figure out how to categorize feature values
        x_axis = [rep_level for rep_level,_ in pred_tups]
        #y_axis = [similarity_to_original_preds(orig, tups) for _, tups in pred_tups]
        y_axis = [disparate_impact(triples[1:], unprotected_group, protected_group) for _,triples in pred_tups]    
        #triples[0] = (Pre-Repaired Feature,Response,Prediction)
        #triples[1] = (WHITE,1,1)
        plt.plot(x_axis, y_axis, label=protected_group)
        features.append(protected_group)
        y_axes.append(y_axis)

      # Format and save the graph to an image file.
      plt.title("Disparate Impact: Pr(Good Outcome given Race = X) / Pr(Good Outcome given Race = WHITE)")
      plt.axis([0,1,0,1.8]) # Make all the plots consistently sized.
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

if __name__=="__main__":
  graph_disparate_impact_accuracy("audits/1455586474.33", "disparate_impact_graphs/DI_Accuracy")
  graph_repair_level_disparate_impact("audits/1455586474.33", "disparate_impact_graphs/RepairLevel_DI")
