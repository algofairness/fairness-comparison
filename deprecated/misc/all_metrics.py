import os, sys
import numpy as np
import utils as ut
import loss_funcs as lf
from algorithms.zafar.fair_classification import *

def zafar_metrics(load_function, sensitive_attr, train_fold_size):
  # fair_classification/*
  #print classifier fairness stats: acc_arr, correlation_dict_arr, cov_dict_arr, s_attr_name
  """ Load the data """
  X, y, x_control = load_function()
  ut.compute_p_rule(x_control[sensitive_attr], y)

  """ Split data into train and test """
  X = ut.add_intercept(X)
  x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)
  loss_function = lf._logistic_loss
  
def kam_metrics():
  # fadm/eval
  return 
   
