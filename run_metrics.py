import numpy as np
import pandas as pd
from algorithms.AbstractAlgorithm import *
from algorithms.feldman.FeldmanAlgorithm import *
from algorithms.kamishima.KamishimaAlgorithm import *
from algorithms.zafar.ZafarAlgorithm import *
from algorithms.gen.GenAlgorithm import *
from algorithms.calders.CaldersAlgorithm import *

def print_res(metric):
  print("Accuracy:", metric.accuracy())
  print("DI Score:", metric.DI_score())
  print("BER:", metric.BER())
  print("BCR:", metric.BCR())
  print("CV Score:", metric.CV_score())

def run_metrics(data, prepare, classifier):
  # Gen
  print("Running Baseline SVM, NB, and LR...")
  params = {}
  algorithm = GenAlgorithm(data, params, prepare, classifier)
  svm_actual, svm_predicted, svm_protected, nb_actual, nb_predicted, nb_protected, lr_actual, lr_predicted, lr_protected = algorithm.run()

  # Calders
  print("Running Calders...")
  params = {}
  algorithm = CaldersAlgorithm(data, params, prepare, classifier)
  c2nb_actual, c2nb_predicted, c2nb_protected = algorithm.run()

  '''
  # Feldman
  print("Running Feldman SVM...")
  params = {}
  algorithm = FeldmanAlgorithm(data, params, prepare, classifier)
  feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected = algorithm.run()
  '''

  # Kamishima
  print("Running Kamishima...")
  params = {}
  params["eta"] = 1
  algorithm = KamishimaAlgorithm(data, params, prepare, classifier)
  kam1_actual, kam1_predicted, kam1_protected = algorithm.run()

  params["eta"] = 30
  algorithm = KamishimaAlgorithm(data, params, prepare, classifier)
  kam30_actual, kam30_predicted, kam30_protected = algorithm.run()

  params["eta"] = 100
  algorithm = KamishimaAlgorithm(data, params, prepare, classifier)
  kam100_actual, kam100_predicted, kam100_protected = algorithm.run()

  params["eta"] = 500
  algorithm = KamishimaAlgorithm(data, params, prepare, classifier)
  kam500_actual, kam500_predicted, kam500_protected = algorithm.run()

  params["eta"] = 1000
  algorithm = KamishimaAlgorithm(data, params, prepare, classifier)
  kam1000_actual, kam1000_predicted, kam1000_protected = algorithm.run()

  # Zafar
  print("Running Zafar...")
  params = {}
  algorithm = ZafarAlgorithm(data, params, prepare, classifier)
  zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected = algorithm.run()

  params["apply_fairness_constraints"] = 1
  params["sensitive_attrs_to_cov_thresh"] = {algorithm.sensitive_attr:0}
  algorithm = ZafarAlgorithm(data, params, prepare, classifier)
  zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected = algorithm.run()

  params["apply_accuracy_constraint"] = 1
  params["apply_fairness_constraints"] = 0
  params["sensitive_attrs_to_cov_thresh"] = {}
  params["gamma"] = 0.5
  algorithm = ZafarAlgorithm(data, params, prepare, classifier)
  zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected = algorithm.run()

  params["sep_constraint"] = 1
  params["gamma"] = 1000.0
  algorithm = ZafarAlgorithm(data, params, prepare, classifier)
  zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected = algorithm.run()
  print("\n")

  # Generate Metric calculators
  svm_metrics = Metrics(svm_actual, svm_predicted, svm_protected)
  print("========================================= SVM ==========================================\n")
  print_res(svm_metrics)
  print("\n")

  nb_metrics = Metrics(nb_actual, nb_predicted, nb_protected)
  print("========================================== NB ==========================================\n")
  print_res(nb_metrics)
  print("\n")

  lr_metrics = Metrics(lr_actual, lr_predicted, lr_protected)
  print("========================================== LR ==========================================\n")
  print_res(lr_metrics)
  print("\n")

  c2nb_metrics = Metrics(c2nb_actual, c2nb_predicted, c2nb_protected)
  print("======================================= Calders ========================================\n")
  print_res(c2nb_metrics)
  print("\n")

  feldman_svm_metrics = Metrics(feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected)
  print("======================================= Feldman ========================================\n")
  print("  Model = SVM: ")
  print_res(feldman_svm_metrics)
  print("\n")

  kam1_metrics = Metrics(kam1_actual, kam1_predicted, kam1_protected)
  print("====================================== Kamishima =======================================\n")
  print("  ETA = 1: ")
  print_res(kam1_metrics)
  print("\n")

  kam30_metrics = Metrics(kam30_actual, kam30_predicted, kam30_protected)
  print("  ETA = 30: ")
  print_res(kam30_metrics)
  print("\n")

  kam100_metrics = Metrics(kam100_actual, kam100_predicted, kam100_protected)
  print("  ETA = 100: ")
  print_res(kam100_metrics)
  print("\n")

  kam500_metrics = Metrics(kam500_actual, kam500_predicted, kam500_protected)
  print("  ETA = 500: ")
  print_res(kam500_metrics)
  print("\n")

  kam1000_metrics = Metrics(kam1000_actual, kam1000_predicted, kam1000_protected)
  print("  ETA = 1000: ")
  print_res(kam1000_metrics)
  print("\n")

  zafar_unconstrained_metrics = Metrics(zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected)
  print("======================================== Zafar =========================================\n")
  print("  Unconstrained: ")
  print_res(zafar_unconstrained_metrics)
  print("\n")

  zafar_opt_accuracy_metrics = Metrics(zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected)
  print("  Optimized for accuracy: ")
  print_res(zafar_opt_accuracy_metrics)
  print("\n")

  zafar_opt_fairness_metrics = Metrics(zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected)
  print("  Optimized for fairness: ")
  print_res(zafar_opt_fairness_metrics)
  print("\n")

  zafar_nopos_classification_metrics = Metrics(zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected)
  print("  No positive classification error: ")
  print_res(zafar_nopos_classification_metrics)

if __name__ == '__main__':
  print("###################################### German Data ######################################\n")
  run_metrics('german', prepare_german, classify_german)
  print("\n")

#  print("###################################### Adult Data #######################################\n")
#  run_metrics('adult')
