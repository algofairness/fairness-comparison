import os,sys
import numpy as np
sys.path.insert(0, 'zafar_fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints



def train_test_classifier(x_train, y_train, x_control_train, x_test, y_test, x_control_test, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma):

  w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
  #W is learned weight vector for the classifier

  #Calculate the accuracy by comparing against correct classification
  train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)

  #Take the dot product of W and each element in the testing set
  distances_boundary_test = (np.dot(x_test, w)).tolist()

  #Classify class labels based off sign (+/-) of result of dot product
  all_class_labels_assigned_test = np.sign(distances_boundary_test)

  correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
  cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
  ut.print_mutual_information(all_class_labels_assigned_test, x_control_test, sensitive_attrs)
  p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])

  return w, p_rule, test_score
