from repairers.GeneralRepairer import Repairer
from loggers import vprint
from measurements import get_conf_matrix
from model_factories.AbstractModelFactory import AbstractModelFactory
from model_factories.AbstractModelVisitor import AbstractModelVisitor

from multiprocessing import Pool, cpu_count
import csv
import time
import os
import json
import gc

ENABLE_MULTIPROCESSING = True
SAVE_REPAIRED_DATA = True
SAVE_PREDICTION_DETAILS = True

# Used to share a copy of the dataset between multiprocessing processes.
shared_all = None
shared_train = None
shared_test = None

def _audit_worker(params):
  global shared_all
  global shared_train
  global shared_test

  model_or_factory, headers, ignored_features, feature_to_repair, repair_level, output_file = params

  index_to_repair = headers.index(feature_to_repair)

  repairer = Repairer(shared_all, index_to_repair,
                      repair_level, features_to_ignore=ignored_features)

  # Build a model on repaired training data if specified.
  if isinstance(model_or_factory, AbstractModelFactory):
    rep_train = repairer.repair(shared_train)
    model = model_or_factory.build(rep_train)

    # Log that this specific model was used for this repair level.
    with open(output_file + ".models.names.txt", "a") as f:
      f.write("{}: {}\n".format(repair_level, model.model_name))

    # Save the repaired version of the data if specified.
    if SAVE_REPAIRED_DATA:
      with open(output_file + ".train.repaired_{}.data".format(repair_level), "w") as f:
        writer = csv.writer(f)
        for row in [headers]+rep_train:
          writer.writerow(row)
  else:
    model = model_or_factory

  rep_test = repairer.repair(shared_test)

  test_name = "{}_{}".format(index_to_repair, repair_level)
  pred_tuples = model.test(rep_test, test_name=test_name)
  conf_table = get_conf_matrix(pred_tuples)

  # Save the repaired version of the data if specified.
  if SAVE_REPAIRED_DATA:
    with open(output_file + ".test.repaired_{}.data".format(repair_level), "w") as f:
      writer = csv.writer(f)
      for row in [headers]+rep_test:
        writer.writerow(row)


  # Save the prediction_tuples and the original values of the features to repair.
  if SAVE_PREDICTION_DETAILS:
    with open(output_file + ".repaired_{}.predictions".format(repair_level), "w") as f:
      writer = csv.writer(f)
      file_headers = ["Pre-Repaired Feature", "Response", "Prediction"]
      writer.writerow(file_headers)
      for i, orig_row in enumerate(shared_test):
        row = [orig_row[index_to_repair], pred_tuples[i][0], pred_tuples[i][1]]
        writer.writerow(row)

  del rep_test
  del repairer
  gc.collect()

  return (repair_level, conf_table)


class GradientFeatureAuditor(object):
  def __init__(self, model_or_factory, headers, train_set, test_set, repair_steps=10,
                features_to_ignore = []):
    self.repair_steps = repair_steps
    self.model_or_factory = model_or_factory
    self.headers = headers
    self.features_to_ignore = features_to_ignore
    self.AUDIT_DIR = "audits"
    self.OUTPUT_DIR = "{}/{}".format(self.AUDIT_DIR, time.time())

    global shared_all
    global shared_train
    global shared_test

    shared_all = train_set + test_set
    shared_test = test_set
    shared_train = train_set

    # Set to `True` to allow the repaired data to be saved to a file.
    # Note: Be cautious when using this on large-sized datasets.

    # Create any output directories that don't exist.
    for directory in [self.AUDIT_DIR, self.OUTPUT_DIR]:
      if not os.path.exists(directory):
        os.makedirs(directory)

  def audit_feature(self, feature_to_repair, output_file):
    repair_increase_per_step = 1.0/self.repair_steps
    repair_level = 0.0

    worker_params = []
    while repair_level <= 1.0:

      call_params = (self.model_or_factory, self.headers, self.features_to_ignore, feature_to_repair, repair_level, output_file)
      worker_params.append( call_params )
      repair_level += repair_increase_per_step

    if ENABLE_MULTIPROCESSING:
      pool = Pool(processes=cpu_count()/2 or 1, maxtasksperchild=1)
      conf_table_tuples = pool.map(_audit_worker, worker_params)
      pool.close()
      pool.join()
      del pool
    else:
      conf_table_tuples = [_audit_worker(params) for params in worker_params]

    conf_table_tuples.sort(key=lambda tuples: tuples[0])

    with open(output_file, "a") as f:
      f.write("GFA Audit for:{}\n".format(feature_to_repair))
      for repair_level, conf_table in conf_table_tuples:
        json_conf_table = json.dumps(conf_table)
        f.write("{}:{}\n".format(repair_level, json_conf_table))

  def audit(self, verbose=False):
    features_to_audit = [h for i, h in enumerate(self.headers) if i not in self.features_to_ignore]

    output_files = []
    for i, feature in enumerate(features_to_audit):
      message = "Auditing: '{}' ({}/{}).".format(feature,i+1,len(features_to_audit))
      vprint(message, verbose)

      cleaned_feature_name = feature.replace(".","_").replace(" ","_")
      output_file = "{}.audit".format(cleaned_feature_name)
      full_filepath = self.OUTPUT_DIR + "/" + output_file
      output_files.append(full_filepath)

      self.audit_feature(feature, full_filepath)

    print "Audit files dumped to: {}".format(self.OUTPUT_DIR)
    return output_files


class MockModel(AbstractModelVisitor):
  def test(self, test_set, arff_prefix="test", response_col=0, test_name=""):
    return [(entry[response_col], entry[response_col]) for entry in test_set]

def test():
  headers = ["response", "duplicate", "constant"]
  train = [[i,i,1] for i in xrange(100)]
  test = train[:] # Copy the training data.
  model = MockModel(test)
  repair_steps = 5
  gfa = GradientFeatureAuditor(model, headers, train, test,
                               repair_steps=repair_steps)
  output_files = gfa.audit()

  print "correct # of audit files produced? --", len(output_files) == len(train[0]) # The number of features.

  with open(output_files[0]) as f:
    print "correct # of lines per file? --", len(f.readlines()) == repair_steps+2 # +1 for the header-line and +1 for the level=0 step.

  files_not_empty = all(os.stat(f).st_size!=0 for f in output_files)
  print "all audit files not empty? --", files_not_empty

  #TODO: Test the optional predictions and repaired output files.

if __name__=="__main__":
  test()
