import numpy as np
import pandas as pd
import sys
from datetime import timedelta
sys.path.insert(0,'algorithms')
from AbstractAlgorithm import *
from feldman.FeldmanAlgorithm import *
from kamishima.KamishimaAlgorithm import *
from zafar.ZafarAlgorithm import *
from gen.GenAlgorithm import *
from calders.CaldersAlgorithm import *

def ret_res(metric):
  return metric.accuracy(), metric.DI_score(), metric.BER(), metric.BCR(), metric.CV_score(), metric.MCC()

def print_res(metric):
  print(("Accuracy:", metric.accuracy()))
  print(("DI Score:", metric.DI_score()))
  print(("BER:", metric.BER()))
  print(("BCR:", metric.BCR()))
  print(("CV Score:", metric.CV_score()))

def run_metrics(data, listoflists, times):
  print("Running algorithms...")
  # Gen
  params = {}
  algorithm = GenAlgorithm(data, params)
  svm_actual, svm_predicted, svm_protected, svm_time, nb_actual, nb_predicted, nb_protected, nb_time, lr_actual, lr_predicted, lr_protected, lr_time = algorithm.run()

  # Calders
  params = {}
  algorithm = CaldersAlgorithm(data, params)
  c2nb_actual, c2nb_predicted, c2nb_protected, c2nb_time = algorithm.run()

  # Feldman
  params = {"model": Weka_SVM}
  algorithm = FeldmanAlgorithm(data, params)
  feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected, feldman_svm_time = algorithm.run()

  params = {"model": Weka_DecisionTree}
  algorithm = FeldmanAlgorithm(data, params)
  feldman_wdt_actual, feldman_wdt_predicted, feldman_wdt_protected, feldman_wdt_time = algorithm.run()

  # Kamishima
  params = {}
  #params["eta"] = 1
  algorithm = KamishimaAlgorithm(data, params)
  kam_actual, kam_predicted, kam_protected, kam_time = algorithm.run()

  '''
  #params["eta"] = 30
  algorithm = KamishimaAlgorithm(data, params)
  kam30_actual, kam30_predicted, kam30_protected, kam30_time = algorithm.run()

  #params["eta"] = 100
  algorithm = KamishimaAlgorithm(data, params)
  kam100_actual, kam100_predicted, kam100_protected, kam100_time = algorithm.run()

  #params["eta"] = 500
  algorithm = KamishimaAlgorithm(data, params)
  kam500_actual, kam500_predicted, kam500_protected, kam500_time = algorithm.run()

  if(data == "ricci"):
    kam1000_actual,kam1000_predicted, kam1000_protected, kam1000_time = [],[],[], 'NA'
  else:
    params["eta"] = 1000
    algorithm = KamishimaAlgorithm(data, params)
    kam1000_actual, kam1000_predicted, kam1000_protected, kam1000_time = algorithm.run()
  '''

  # Zafar
  params = {}
  algorithm = ZafarAlgorithm(data, params)
  zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected, zafar_unconstrained_time = algorithm.run()

  params["apply_fairness_constraints"] = 1
  params["sensitive_attrs_to_cov_thresh"] = {algorithm.sensitive_attr:0}
  algorithm = ZafarAlgorithm(data, params)
  zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected, zafar_opt_accuracy_time = algorithm.run()

  params["apply_accuracy_constraint"] = 1
  params["apply_fairness_constraints"] = 0
  params["sensitive_attrs_to_cov_thresh"] = {}
  params["gamma"] = 0.5
  algorithm = ZafarAlgorithm(data, params)
  zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected, zafar_opt_fairness_time = algorithm.run()

  if(data == "german"):
    zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected, zafar_nopos_classification_time = [], [], [], 'NA'
  else:
    params["sep_constraint"] = 1
    params["gamma"] = 1000.0
    algorithm = ZafarAlgorithm(data, params)
    zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected, zafar_nopos_classification_time = algorithm.run()
#  print("\n")

  # Generate Metric calculators
  svm_metrics = Metrics(svm_actual, svm_predicted, svm_protected)
#  print("========================================= SVM ==========================================\n")
  # print_res(svm_metrics)
  results = ret_res(svm_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][0].append(results[i])
  times[0].append(svm_time)
      
#  print("\n")

  nb_metrics = Metrics(nb_actual, nb_predicted, nb_protected)
#  print("========================================== NB ==========================================\n")
  #print_res(nb_metrics)
  results = ret_res(nb_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][1].append(results[i])
  times[1].append(nb_time)

#  print("\n")

  lr_metrics = Metrics(lr_actual, lr_predicted, lr_protected)
#  print("========================================== LR ==========================================\n")
  #print_res(lr_metrics)
  results = ret_res(lr_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][2].append(results[i])
  times[2].append(lr_time)

# print("\n")

  c2nb_metrics = Metrics(c2nb_actual, c2nb_predicted, c2nb_protected)
#  print("======================================= Calders ========================================\n")
  #print_res(c2nb_metrics)
  results = ret_res(c2nb_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][3].append(results[i])
  times[3].append(c2nb_time)

#  print("\n")

  feldman_svm_metrics = Metrics(feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected)
  feldman_wdt_metrics = Metrics(feldman_wdt_actual, feldman_wdt_predicted, feldman_wdt_protected)
#  print("======================================= Feldman ========================================\n")
#  print("  Model = SVM: ")
#  print_res(feldman_svm_metrics)
  results = ret_res(feldman_svm_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][4].append(results[i])
  times[4].append(feldman_svm_time)

  results = ret_res(feldman_wdt_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][5].append(results[i])
  times[5].append(feldman_wdt_time)

#  print("\n")
   

  kam_metrics = Metrics(kam_actual, kam_predicted, kam_protected)
#  print("====================================== Kamishima =======================================\n")
#  print("  ETA = 1: ")
  #print_res(kam1_metrics)
  results = ret_res(kam_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][6].append(results[i])
  times[6].append(kam_time)

#  print("\n")

  '''
  kam30_metrics = Metrics(kam30_actual, kam30_predicted, kam30_protected)
#  print("  ETA = 30: ")
  #print_res(kam30_metrics)
  results = ret_res(kam30_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][7].append(results[i])
  times[7].append(kam30_time)

#  print("\n")

  kam100_metrics = Metrics(kam100_actual, kam100_predicted, kam100_protected)
#  print("  ETA = 100: ")
  #print_res(kam100_metrics)
  results = ret_res(kam100_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][8].append(results[i])
  times[8].append(kam100_time)

#  print("\n")

  kam500_metrics = Metrics(kam500_actual, kam500_predicted, kam500_protected)
#  print("  ETA = 500: ")
  #print_res(kam500_metrics)
  results = ret_res(kam500_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][9].append(results[i])
  times[9].append(kam500_time)

#  print("\n")

  kam1000_metrics = Metrics(kam1000_actual, kam1000_predicted, kam1000_protected)
#  print("  ETA = 1000: ")
  #print_res(kam1000_metrics)
  results = ret_res(kam1000_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][10].append(results[i])
  times[10].append(kam1000_time)

#  print("\n")
  '''

  zafar_unconstrained_metrics = Metrics(zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected)
#  print("======================================== Zafar =========================================\n")
#  print("  Unconstrained: ")
  #print_res(zafar_unconstrained_metrics)
  results = ret_res(zafar_unconstrained_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][7].append(results[i])
  times[7].append(zafar_unconstrained_time)

#  print("\n")

  zafar_opt_accuracy_metrics = Metrics(zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected)
#  print("  Optimized for accuracy: ")
  #print_res(zafar_opt_accuracy_metrics)
  results = ret_res(zafar_opt_accuracy_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][8].append(results[i])
  times[8].append(zafar_opt_accuracy_time)

#  print("\n")

  zafar_opt_fairness_metrics = Metrics(zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected)
#  print("  Optimized for fairness: ")
  #print_res(zafar_opt_fairness_metrics)
  results = ret_res(zafar_opt_fairness_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][9].append(results[i])
  times[9].append(zafar_opt_fairness_time)

#  print("\n")

  zafar_nopos_classification_metrics = Metrics(zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected)
#  print("  No positive classification error: ")
  #print_res(zafar_nopos_classification_metrics)
  results = ret_res(zafar_nopos_classification_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][10].append(results[i])
  times[10].append(zafar_nopos_classification_time)


def get_sd(vals_per_split, mean):
  less_mean = []
  for i in vals_per_split:
    less_mean.append((i-mean)**2)  
  mean_sq_diffs = sum(less_mean[0:len(less_mean)])/len(less_mean)
  return np.sqrt(mean_sq_diffs)
  

def run_repeatedly(data, runs=10):
  acc, final_acc = [[] for i in range(11)], [] 
  di, final_di = [[] for i in range(11)], [] 
  ber, final_ber = [[] for i in range(11)], [] 
  bcr, final_bcr = [[] for i in range(11)], [] 
  cv, final_cv = [[] for i in range(11)], []
  auc, final_auc = [[] for i in range(11)], []
  mcc, final_mcc = [[] for i in range(11)], []
  tim, final_time = [[] for i in range(11)], []
  metrics = [acc,di,ber,bcr,cv,mcc]
  final_metrics = [final_acc,final_di,final_ber,final_bcr,final_cv,final_mcc]
  sd = [[[0] for i in range(11)],[[0] for i in range(11)],[[0] for i in range(11)],[[0] for i in range(11)],[[0] for i in range(11)],[[0] for i in range(11)]]

  for i in range(0,runs):
    run_metrics(data, metrics, tim)

  for i in range(0,len(tim)):
    if 'NA' in tim[i]:
      tim[i] = [k for k in tim[i] if k != 'NA']
    if len(tim[i]) == 0:
      final_time.append('NA')
    else:
      mean_time = sum(tim[i], timedelta()) / len(tim[i])
      final_time.append(str(mean_time))

  for i in range(0,len(metrics)):
    # x is list of lists where each list is an algorithm's runs
    for x in range(len(metrics[i])):
      if 'NA' in metrics[i][x]:
        metrics[i][x] = [k for k in metrics[i][x] if k != 'NA']
      if len(metrics[i][x]) == 0:
        final_metrics[i].append('NA') 
        sd[i][x] = 'NA'
      else:
        mean = sum(metrics[i][x][0:len(metrics[i][x])])/len(metrics[i][x]) 
        sd[i][x] = get_sd(metrics[i][x], mean)
        final_metrics[i].append(mean)

  # Create DataFrame of results and export to csv located in results directory
  export_to = 'results/' + data + '.csv' 
  headers = ['Algorithms','Acc', 'Acc_SD', 'DI', 'DI_SD','BER', 'BER_SD', 'BCR', 'BCR_SD', 'CV', 'CV_SD','MCC','MCC_SD', 'Run Time']
  #algorithms = ['SVM','NB','LR','Calders','Kamishima','Zafar Unconstrained','Zafar w Accuracy Constraint','Zafar w Fairness Constraint','Zafar No Pos Misclassification']
  algorithms = ['SVM','NB','LR','Calders','Feldman SVM', 'Feldman WDT','Kamishima','Zafar Unconstrained','Zafar w Accuracy Constraint','Zafar w Fairness Constraint','Zafar No Pos Misclassification']
  #algorithms = ['SVM','NB','LR','Calders','Feldman SVM', 'Feldman WDT','Kamishima eta=1','Kamishima eta=30','Kamishima eta=100','Kamishima eta=500','Kamishima eta=1000','Zafar Unconstrained','Zafar w Accuracy Constraint','Zafar w Fairness Constraint','Zafar No Pos Misclassification']

  d = {'Algorithms':algorithms,'Acc':final_acc, 'Acc_SD':sd[0] , 'DI':final_di, 'DI_SD':sd[1] , 'BER':final_ber, 'BER_SD':sd[2] , 'BCR':final_bcr, 'BCR_SD':sd[3] , 'CV':final_cv, 'CV_SD':sd[4], 'MCC':final_mcc, 'MCC_SD':sd[5], 'Run Time':final_time}
  df = pd.DataFrame(data=d)
  df = df[headers]
  df.to_csv(export_to, index=False) 

if __name__ == '__main__':
  print('Analyzing German data...')
  run_repeatedly('german')
  print('Complete.')
  print("\n")

  '''
  print('Analyzing Ricci data...')
  run_repeatedly("ricci")
  print('Complete.')
  print("\n")

  print('Analyzing Adult data...')
  run_repeatedly('adult')
  print('Complete.')
  print("\n")
  
  print('Analyzing Retailer data...')
  run_repeatedly("retailer")
  print('Complete.')

  print('Analyzing Small Retailer data...')
  run_repeatedly("small-retailer",1)
  print('Complete.')
  '''
