from data.adult.load_adult import *
from preprocessing.deprecated.black_box_auditing import *
import numpy as np
import algorithms.zafar.fair_classification.utils as ut
from random import shuffle

def prepare_adult():
  sensitive_attrs = ["sex"]
  sensitive_attr = sensitive_attrs[0]
  train_fold_size = 0.66

  run_adult_repair()

  X, y, x_control = load_adult_data("data/adult/adult-all-numerical-converted.csv")
  X = ut.add_intercept(X)
  perm = list(range(0, len(y)))
  shuffle(perm)
  X = X[perm]
  y = y[perm]

  for k in x_control.keys():
    x_control[k] = x_control[k][perm]

  x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

  y_train_fixed = []
  for y in y_train:
    if y == -1.0:
      y_train_fixed.append(0.0)
    elif y == 1.0:
      y_train_fixed.append(1.0)

  y_test_fixed = []
  for y in y_test:
    if y == -1.0:
      y_test_fixed.append(0.0)
    elif y == 1.0:
      y_test_fixed.append(1.0)

  x_control_train_fixed_val = []
  for x in x_control_train[sensitive_attr]:
    if x == 0.0:
      x_control_train_fixed_val.append(0.0)
    elif x == 1.0:
      x_control_train_fixed_val.append(1.0)
  x_control_train[sensitive_attr] = np.array(x_control_train_fixed_val)

  x_control_test_fixed_val = []
  for x in x_control_test[sensitive_attr]:
    if x == 0.0:
      x_control_test_fixed_val.append(0.0)
    elif x == 1.0:
      x_control_test_fixed_val.append(1.0)
  x_control_test[sensitive_attr] = np.array(x_control_test_fixed_val)

  # Change types to run metrics
  x_control_train[sensitive_attr] = [int(i) for i in x_control_train[sensitive_attr]]
  x_control_test[sensitive_attr] = [int(i) for i in x_control_test[sensitive_attr]]

  return x_train, np.array(y_train_fixed), x_control_train, x_test, np.array(y_test_fixed), x_control_test, sensitive_attr
