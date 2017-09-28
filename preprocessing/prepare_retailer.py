from data.retailer.load_data import *
from black_box_auditing import *
import numpy as np
import algorithms.zafar.fair_classification.utils as ut
from random import shuffle

def prepare_retailer():
  sensitive_attrs = ["urace_orig"]
  sensitive_attr = sensitive_attrs[0]
  train_fold_size = 2.0/3.0

  run_retailer_repair()

  X, y, x_control = load_retailer_data()
  
  perm = range(0, len(y))
  shuffle(perm)
  X = X[perm]
  y = y[perm]
  x_control["urace_orig"] = np.array(x_control["urace_orig"])

  for k in x_control.keys():
    x_control[k] = x_control[k][perm]

  x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

  x_control_train["urace_orig"] = np.array(x_control_train["urace_orig"])
  x_control_test["urace_orig"] = np.array(x_control_test["urace_orig"])

  # Change types to run metrics
  x_train = x_train.astype(float)
  y_train = y_train.astype(float)
  x_test = x_test.astype(float)
  y_test = y_test.astype(float)
  x_control_train[sensitive_attr] = x_control_train[sensitive_attr].astype(float)
  x_control_test[sensitive_attr] = x_control_test[sensitive_attr].astype(float)

  return x_train, y_train, x_control_train, x_test, y_test, x_control_test, sensitive_attr
