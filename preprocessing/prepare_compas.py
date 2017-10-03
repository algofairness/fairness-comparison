from data.propublica.load_numerical_compas import *
from preprocessing.black_box_auditing import *
import numpy as np
import algorithms.zafar.fair_classification.utils as ut
from random import shuffle

def prepare_compas():
  sensitive_attrs = ["race"]
  sensitive_attr = sensitive_attrs[0]
  train_fold_size = 0.7

  run_compas_repair()

  X, y, x_control = load_compas_data("all_numeric.csv")

  perm = list(range(0, len(y))) # shuffle data before creating each fold
  shuffle(perm)
  X = X[perm]
  y = y[perm]

  for k in x_control.keys():
    x_control[k] = x_control[k][perm]

  # Split into train and test
  x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

  swapped_x = []
  for i in x_control_train[sensitive_attr]:
    if i == 0:
      swapped_x.append(1)
    if i == 1:
      swapped_x.append(0)
  x_control_train[sensitive_attr] = swapped_x

  swapped_x = []
  for i in x_control_test[sensitive_attr]:
    if i == 0:
      swapped_x.append(1)
    if i == 1:
      swapped_x.append(0)
  x_control_test[sensitive_attr] = swapped_x

  return x_train, np.array(y_train), x_control_train, x_test, np.array(y_test), x_control_test, sensitive_attr
