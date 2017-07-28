import numpy as np
import pandas as pd
from algorithms.AbstractAlgorithm import *
from algorithms.feldman.FeldmanAlgorithm import *
from algorithms.kamishima.KamishimaAlgorithm import *
from algorithms.zafar.ZafarAlgorithm import *

def print_res(metric):
  print("Accuracy:", metric.accuracy())
  print("DI Score:", metric.DI_score())
  print("BER:", metric.BER())
  print("BCR:", metric.BCR())
  print("CV Score:", metric.CV_score())

def print_metrics(data):
  # Feldman
  print("Running Feldman SVM...")
  params = {}
  algorithm = FeldmanAlgorithm(data, params)
  feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected = algorithm.run()

  # Kamishima
  print("Running Kamishima...")
  params = {}
  params["eta"] = 1
  algorithm = KamishimaAlgorithm(data, params)
  kam1_actual, kam1_predicted, kam1_protected = algorithm.run()

  params["eta"] = 30
  algorithm = KamishimaAlgorithm(data, params)
  kam30_actual, kam30_predicted, kam30_protected = algorithm.run()

  params["eta"] = 100
  algorithm = KamishimaAlgorithm(data, params)
  kam100_actual, kam100_predicted, kam100_protected = algorithm.run()

  params["eta"] = 500
  algorithm = KamishimaAlgorithm(data, params)
  kam500_actual, kam500_predicted, kam500_protected = algorithm.run()

  params["eta"] = 1000
  algorithm = KamishimaAlgorithm(data, params)
  kam1000_actual, kam1000_predicted, kam1000_protected = algorithm.run()

  # Zafar
  print("Running Zafar...")


  kam1_metrics = Metrics(kam1_actual, kam1_predicted, kam1_protected)
  kam30_metrics = Metrics(kam30_actual, kam30_predicted, kam30_protected)
  kam100_metrics = Metrics(kam100_actual, kam100_predicted, kam100_protected)
  kam500_metrics = Metrics(kam500_actual, kam500_predicted, kam500_protected)
  kam1000_metrics = Metrics(kam1000_actual, kam1000_predicted, kam1000_protected)

  print("====================================== Kamishima =======================================")   
  print("  ETA = 1: ")
  print_res(kam1_metrics)
  print("\n")
  print("  ETA = 30: ")
  print_res(kam30_metrics)
  print("\n")
  print("  ETA = 100: ")
  print_res(kam100_metrics)
  print("\n")
  print("  ETA = 500: ")
  print_res(kam500_metrics)
  print("\n")
  print("  ETA = 1000: ")
  print_res(kam1000_metrics)
  print("\n")
  
if __name__ == '__main__':
  print("###################################### German Data ######################################")
  print_metrics('german')
  print("\n")

  print("###################################### Adult Data #######################################")
  print_metrics('adult')
  print("\n")

