from data.german.load_german_data import *
from misc.black_box_auditing import *
import numpy as np
import algorithms.zafar.fair_classification.utils as ut
from random import shuffle

def prepare_data():
  sensitive_attrs = ["sex"]
  sensitive_attr = sensitive_attrs[0]
  train_fold_size = 0.3

  run_german_repair()

  X, y, x_control = load_german_data("german_numeric_sex_encoded_fixed.csv")

  perm = range(0, len(y))
  shuffle(perm)
  X = X[perm]

  y = y[perm]

  x_control["sex"] = np.array(x_control["sex"])

  for k in x_control.keys():
    x_control[k] = x_control[k][perm]

  x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

  x_control_train["sex"] = np.array(x_control_train["sex"])
  x_control_test["sex"] = np.array(x_control_test["sex"])

  # Change types to run metrics
  x_train = x_train.astype(float)
  y_train = y_train.astype(float)
  x_test = x_test.astype(float)
  y_test = y_test.astype(float)
  x_control_train[sensitive_attr] = x_control_train[sensitive_attr].astype(float)
  x_control_test[sensitive_attr] = x_control_test[sensitive_attr].astype(float)
