import numpy as np
import pandas as pd
from algorithms.AbstractAlgorithm import *
from algorithms.feldman.FeldmanAlgorithm import *
from algorithms.kamishima.KamishimaAlgorithm import *
from algorithms.zafar.ZafarAlgorithm import *
from algorithms.gen.GenAlgorithm import *
from algorithms.calders.CaldersAlgorithm import *

def ret_res(metric):
  return metric.accuracy(), metric.DI_score(), metric.BER(), metric.BCR(), metric.CV_score()

def print_res(metric):
  print("Accuracy:", metric.accuracy())
  print("DI Score:", metric.DI_score())
  print("BER:", metric.BER())
  print("BCR:", metric.BCR())
  print("CV Score:", metric.CV_score())

def run_metrics(data, listoflists):
  print("Running algorithms...")
  # Gen
#  print("Running Baseline SVM, NB, and LR...")
  params = {}
  algorithm = GenAlgorithm(data, params)
  svm_actual, svm_predicted, svm_protected, nb_actual, nb_predicted, nb_protected, lr_actual, lr_predicted, lr_protected = algorithm.run()

  # Calders
#  print("Running Calders...")
  params = {}
  algorithm = CaldersAlgorithm(data, params)
  c2nb_actual, c2nb_predicted, c2nb_protected = algorithm.run()

   
  # Feldman
  print("Running Feldman SVM...")
  params = {}
  algorithm = FeldmanAlgorithm(data, params)
  feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected = algorithm.run()
   

  # Kamishima
#  print("Running Kamishima...")
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
#  print("Running Zafar...")
  params = {}
  algorithm = ZafarAlgorithm(data, params)
  zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected = algorithm.run()

  params["apply_fairness_constraints"] = 1
  params["sensitive_attrs_to_cov_thresh"] = {algorithm.sensitive_attr:0}
  algorithm = ZafarAlgorithm(data, params)
  zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected = algorithm.run()

  params["apply_accuracy_constraint"] = 1
  params["apply_fairness_constraints"] = 0
  params["sensitive_attrs_to_cov_thresh"] = {}
  params["gamma"] = 0.5
  algorithm = ZafarAlgorithm(data, params)
  zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected = algorithm.run()

  params["sep_constraint"] = 1
  params["gamma"] = 1000.0
  algorithm = ZafarAlgorithm(data, params)
  zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected = algorithm.run()
#  print("\n")

  # Generate Metric calculators
  svm_metrics = Metrics(svm_actual, svm_predicted, svm_protected)
#  print("========================================= SVM ==========================================\n")
  # print_res(svm_metrics)
  results = ret_res(svm_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][0].append(results[i])
      
#  print("\n")

  nb_metrics = Metrics(nb_actual, nb_predicted, nb_protected)
#  print("========================================== NB ==========================================\n")
  #print_res(nb_metrics)
  results = ret_res(nb_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][1].append(results[i])

#  print("\n")

  lr_metrics = Metrics(lr_actual, lr_predicted, lr_protected)
#  print("========================================== LR ==========================================\n")
  #print_res(lr_metrics)
  results = ret_res(lr_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][2].append(results[i])

#  print("\n")

  c2nb_metrics = Metrics(c2nb_actual, c2nb_predicted, c2nb_protected)
#  print("======================================= Calders ========================================\n")
  #print_res(c2nb_metrics)
  results = ret_res(c2nb_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][3].append(results[i])

#  print("\n")

   
  feldman_svm_metrics = Metrics(feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected)
#  print("======================================= Feldman ========================================\n")
#  print("  Model = SVM: ")
#  print_res(feldman_svm_metrics)
  #results = ret_res(feldman_svm_metrics)
  results = [0,0,0,0,0]
  for i in range(0,len(listoflists)):
    listoflists[i][4].append(results[i])

#  print("\n")
   

  kam1_metrics = Metrics(kam1_actual, kam1_predicted, kam1_protected)
#  print("====================================== Kamishima =======================================\n")
#  print("  ETA = 1: ")
  #print_res(kam1_metrics)
  results = ret_res(kam1_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][5].append(results[i])

#  print("\n")

  kam30_metrics = Metrics(kam30_actual, kam30_predicted, kam30_protected)
#  print("  ETA = 30: ")
  #print_res(kam30_metrics)
  results = ret_res(kam30_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][6].append(results[i])

#  print("\n")

  kam100_metrics = Metrics(kam100_actual, kam100_predicted, kam100_protected)
#  print("  ETA = 100: ")
  #print_res(kam100_metrics)
  results = ret_res(kam100_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][7].append(results[i])

#  print("\n")

  kam500_metrics = Metrics(kam500_actual, kam500_predicted, kam500_protected)
#  print("  ETA = 500: ")
  #print_res(kam500_metrics)
  results = ret_res(kam500_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][8].append(results[i])

#  print("\n")

  kam1000_metrics = Metrics(kam1000_actual, kam1000_predicted, kam1000_protected)
#  print("  ETA = 1000: ")
  #print_res(kam1000_metrics)
  results = ret_res(kam1000_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][9].append(results[i])

#  print("\n")

  zafar_unconstrained_metrics = Metrics(zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected)
#  print("======================================== Zafar =========================================\n")
#  print("  Unconstrained: ")
  #print_res(zafar_unconstrained_metrics)
  results = ret_res(zafar_unconstrained_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][10].append(results[i])

#  print("\n")

  zafar_opt_accuracy_metrics = Metrics(zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected)
#  print("  Optimized for accuracy: ")
  #print_res(zafar_opt_accuracy_metrics)
  results = ret_res(zafar_opt_accuracy_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][11].append(results[i])

#  print("\n")

  zafar_opt_fairness_metrics = Metrics(zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected)
#  print("  Optimized for fairness: ")
  #print_res(zafar_opt_fairness_metrics)
  results = ret_res(zafar_opt_fairness_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][12].append(results[i])

#  print("\n")

  zafar_nopos_classification_metrics = Metrics(zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected)
#  print("  No positive classification error: ")
  #print_res(zafar_nopos_classification_metrics)
  results = ret_res(zafar_nopos_classification_metrics)
  for i in range(0,len(listoflists)):
    listoflists[i][13].append(results[i])

def get_sd(vals_per_split, mean):
  less_mean = []
  for i in vals_per_split:
    less_mean.append((i-mean)**2)  
  mean_sq_diffs = sum(less_mean[0:len(less_mean)])/len(less_mean)
  return np.sqrt(mean_sq_diffs)
  

def run_repeatedly(data, runs=10):
  acc, final_acc = [[] for i in range(14)], []
  di, final_di = [[] for i in range(14)], []
  ber, final_ber = [[] for i in range(14)], []
  bcr, final_bcr = [[] for i in range(14)], []
  cv, final_cv = [[] for i in range(14)], []
  metrics = [acc,di,ber,bcr,cv]
  final_metrics = [final_acc,final_di,final_ber,final_bcr,final_cv]
  sd = [0,0,0,0,0]

  for i in range(0,runs):
    run_metrics(data, metrics)

  for i in range(0,len(metrics)):
    for x in metrics[i]:
      if 'NA' in x:
        x = [k for k in x if k != 'NA']
      if len(x) == 0:
        final_metrics[i].append('NA') 
      else:
        mean = sum(x[0:len(x)])/len(x) 
	sd[i] = get_sd(x,mean)
        final_metrics[i].append(mean)

  '''
  final_acc = []
  for x in acc:
    final_acc.append(sum(x[0:len(x)])/runs)

  final_di = []
  for x in di:
    final_di.append(sum(x[0:len(x)])/runs)

  final_ber = []
  for x in ber:
    final_ber.append(sum(x[0:len(x)])/runs)

  final_bcr = []
  for x in bcr:
    final_bcr.append(sum(x[0:len(x)])/runs)

  final_cv = []
  for x in cv:
    final_cv.append(sum(x[0:len(x)])/runs)

  print "ACC: ", final_acc
  print "DI: ", final_di
  print "BER: ", final_ber
  print "BCR: ", final_bcr
  print "CV: ", final_cv
  '''
  
  # Create DataFrame of results and export to csv located in results directory
  export_to = 'results/' + data + '.csv' 
  headers = ['Algorithms','Acc','DI','BER','BCR','CV']
  algorithms = ['SVM','NB','LR','Calders','Feldman','Kamishima eta=1','Kamishima eta=30','Kamishima eta=100','Kamishima eta=500','Kamishima eta=1000','Zafar Unconstrained','Zafar w Accuracy Constraint','Zafar w Fairness Constraint','Zafar No Pos Misclassification']
  #algorithms = ['SVM','NB','LR','Calders','Kamishima eta=1','Kamishima eta=30','Kamishima eta=100','Kamishima eta=500','Kamishima eta=1000','Zafar Unconstrained','Zafar w Accuracy Constraint','Zafar w Fairness Constraint','Zafar No Pos Misclassification']

  d = {'Algorithms':algorithms,'Acc':final_acc,'DI':final_di,'BER':final_ber,'BCR':final_bcr,'CV':final_cv}
  df = pd.DataFrame(data=d)
  df = df[headers]
  df.loc[-1] = ['SD'] + sd
  df.to_csv(export_to) 

if __name__ == '__main__':
  ''' 
  print('Analyzing German data...')
  run_repeatedly('german',1)
  print('Complete.')
  print("\n")

   
  print('Analyzing Adult data...')
  run_repeatedly('adult')
  print('Complete.')
  print("\n")

  print('Analyzing Retailer data...')
  run_repeatedly("retailer",1)
  print('Complete.')
  ''' 

  print('Analyzing Ricci data...')
  run_repeatedly("ricci")
  print('Complete.')
