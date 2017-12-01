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
  return metric.accuracy(), metric.BCR(), metric.MCC(), metric.DI_score(), metric.CV_score()

def print_res(metric):
  print(("Accuracy:", metric.accuracy()))
  print(("DI Score:", metric.DI_score()))
  print(("BER:", metric.BER()))
  print(("BCR:", metric.BCR()))
  print(("CV Score:", metric.CV_score()))

def get_sd(vals_per_split, mean):
  less_mean = []
  for i in vals_per_split:
    less_mean.append((i-mean)**2)
  mean_sq_diffs = sum(less_mean[0:len(less_mean)])/len(less_mean)
  return np.sqrt(mean_sq_diffs)


def run_gen(data, times):
  metrics = [[],[],[]]
  time = [[],[],[]]
  sd = [[],[],[]] 
  final_time = []
  final_metrics = [[],[],[]]

  for i in range(times):
    params = {}
    algorithm = GenAlgorithm(data, params)
    svm_actual, svm_predicted, svm_protected, svm_time, nb_actual, nb_predicted, nb_protected, nb_time, lr_actual, lr_predicted, lr_protected, lr_time = algorithm.run()

    svm_metrics = Metrics(svm_actual, svm_predicted, svm_protected)
    metrics[0].append(ret_res(svm_metrics))
    time[0].append(svm_time)
    nb_metrics = Metrics(nb_actual, nb_predicted, nb_protected)
    metrics[1].append(ret_res(nb_metrics))
    time[1].append(nb_time)
    lr_metrics = Metrics(lr_actual, lr_predicted, lr_protected)
    metrics[2].append(ret_res(lr_metrics))
    time[2].append(lr_time)

  collected_metrics = [[],[],[]]
  for x in range(len(metrics)):  
    for i in range(5):
      m = []
      for j in range(len(metrics[x])):
        m.append(metrics[x][j][i])
      collected_metrics[x].append(m) 

  for i in range(0,len(time)):
    if 'NA' in time[i]:
      time[i] = [k for k in time[i] if k != 'NA']
    if len(time[i]) == 0:
      final_time.append('0:00:00.00')
    else:
      mean_time = sum(time[i], timedelta()) / len(time[i])
      final_time.append(str(mean_time))

  for i in range(len(collected_metrics)):
    # x is list of lists where each list is a metric 
    for x in range(len(collected_metrics[i])):
      if 'NA' in collected_metrics[i][x]:
        collected_metrics[i][x] = [k for k in collected_metrics[i][x] if k != 'NA']
      if len(collected_metrics[i][x]) == 0:
        final_metrics[i].append('NA')
        sd[i].append('NA')
      else:
        mean = sum(collected_metrics[i][x][0:len(collected_metrics[i][x])])/len(collected_metrics[i][x])
        sd[i].append(get_sd(collected_metrics[i][x], mean))
        final_metrics[i].append(mean)
  
  for i in range(len(final_metrics)):
    final_metrics[i].append(final_time[i])
    final_metrics[i].insert(1,sd[i][0])
    final_metrics[i].insert(3,sd[i][1])
    final_metrics[i].insert(5,sd[i][2])
    final_metrics[i].insert(7,sd[i][3])
    final_metrics[i].insert(9,sd[i][4])

  final_metrics[0].insert(0,'SVM')
  final_metrics[1].insert(0,'NB')
  final_metrics[2].insert(0,'LR')

  svm = ','.join(map(str,final_metrics[0]))
  nb = ','.join(map(str,final_metrics[1]))
  lr = ','.join(map(str,final_metrics[2]))
  return svm, nb, lr 

def run_calders(data, times):
  metrics = []
  time = []
  sd = []
  final_metrics = []
  final_time = '' 

  for i in range(times):
    params = {}
    algorithm = CaldersAlgorithm(data, params)
    c2nb_actual, c2nb_predicted, c2nb_protected, c2nb_time = algorithm.run()
    c2nb_metrics = Metrics(c2nb_actual, c2nb_predicted, c2nb_protected)
    metrics.append(ret_res(c2nb_metrics))
    time.append(c2nb_time)

  collected_metrics = []
  for j in range(5):
    m = []
    for i in range(len(metrics)):
      m.append(metrics[i][j])
    collected_metrics.append(m)

  for i in range(0,len(time)):
    if 'NA' in time:
      times = [k for k in times if k != 'NA']
    if len(time) == 0:
      final_time.append('0:00:00.00')
    else:
      mean_time = sum(time, timedelta()) / len(time)
      final_time = str(mean_time)

  for i in range(len(collected_metrics)):
    if 'NA' in collected_metrics[i]:
      collected_metrics[i] = [k for k in collected_metrics[i] if k != 'NA']
    if len(collected_metrics[i]) == 0:
      final_metrics.append('NA')
      sd.append('NA')
    else:
      mean = sum(collected_metrics[i][0:len(collected_metrics[i])])/len(collected_metrics[i])
      sd.append(get_sd(collected_metrics[i], mean))
      final_metrics.append(mean)

  final_metrics.append(final_time)
  final_metrics.insert(1,sd[0])
  final_metrics.insert(3,sd[1])
  final_metrics.insert(5,sd[2])
  final_metrics.insert(7,sd[3])
  final_metrics.insert(9,sd[4])

  final_metrics.insert(0,'Calders')

  c2nb = ','.join(map(str,final_metrics))

  return c2nb 

def run_feldman(data, times):
  metrics = [[],[]]
  time = [[],[]]
  sd = [[],[]]
  final_time = []
  final_metrics = [[],[]]

  for i in range(times):
    params = {"model": Weka_SVM}
    algorithm = FeldmanAlgorithm(data, params)
    feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected, feldman_svm_time = algorithm.run()

    params = {"model": Weka_DecisionTree}
    algorithm = FeldmanAlgorithm(data, params)
    feldman_wdt_actual, feldman_wdt_predicted, feldman_wdt_protected, feldman_wdt_time = algorithm.run()

    feldman_svm_metrics = Metrics(feldman_svm_actual, feldman_svm_predicted, feldman_svm_protected)
    metrics[0].append(ret_res(feldman_svm_metrics))
    time[0].append(feldman_svm_time)
    feldman_dt_metrics = Metrics(feldman_wdt_actual, feldman_wdt_predicted, feldman_wdt_protected)
    metrics[1].append(ret_res(feldman_dt_metrics))
    time[1].append(feldman_wdt_time)

  collected_metrics = [[],[]]
  for x in range(len(metrics)):
    for i in range(5):
      m = []
      for j in range(len(metrics[x])):
        m.append(metrics[x][j][i])
      collected_metrics[x].append(m)

  for i in range(0,len(time)):
    if 'NA' in time[i]:
      time[i] = [k for k in time[i] if k != 'NA']
    if len(time[i]) == 0:
      final_time.append('0:00:00.00')
    else:
      mean_time = sum(time[i], timedelta()) / len(time[i])
      final_time.append(str(mean_time))

  for i in range(len(collected_metrics)):
    # x is list of lists where each list is a metric 
    for x in range(len(collected_metrics[i])):
      if 'NA' in collected_metrics[i][x]:
        collected_metrics[i][x] = [k for k in collected_metrics[i][x] if k != 'NA']
      if len(collected_metrics[i][x]) == 0:
        final_metrics[i].append('NA')
        sd[i].append('NA')
      else:
        mean = sum(collected_metrics[i][x][0:len(collected_metrics[i][x])])/len(collected_metrics[i][x])
        sd[i].append(get_sd(collected_metrics[i][x], mean))
        final_metrics[i].append(mean)

  for i in range(len(final_metrics)):
    final_metrics[i].append(final_time[i])
    final_metrics[i].insert(1,sd[i][0])
    final_metrics[i].insert(3,sd[i][1])
    final_metrics[i].insert(5,sd[i][2])
    final_metrics[i].insert(7,sd[i][3])
    final_metrics[i].insert(9,sd[i][4])

  final_metrics[0].insert(0,'Feldman SVM')
  final_metrics[1].insert(0,'Feldman DT')

  feldman_svm = ','.join(map(str,final_metrics[0]))
  feldman_dt = ','.join(map(str,final_metrics[1]))
  
  return feldman_svm, feldman_dt 

def run_kamishima(data, times=10):
  metrics = [[],[],[]]
  time = [[],[],[]]
  sd = [[],[],[]]
  final_time = []
  final_metrics = [[],[],[]]

  for i in range(times):
    params = {}
    params['var'] = 1
    algorithm = KamishimaAlgorithm(data, params)
    kamboth_actual, kamboth_predicted, kamboth_protected, kamboth_time = algorithm.run()

    params['var'] = 2
    algorithm = KamishimaAlgorithm(data, params)
    kamacc_actual, kamacc_predicted, kamacc_protected, kamacc_time = algorithm.run()

    params['var'] = 3
    algorithm = KamishimaAlgorithm(data, params)
    kamDI_actual, kamDI_predicted, kamDI_protected, kamDI_time = algorithm.run()

    kamboth_metrics = Metrics(kamboth_actual, kamboth_predicted, kamboth_protected)
    metrics[0].append(ret_res(kamboth_metrics))
    time[0].append(kamboth_time)
    kamacc_metrics = Metrics(kamacc_actual, kamacc_predicted, kamacc_protected)
    metrics[1].append(ret_res(kamacc_metrics))
    time[1].append(kamacc_time)
    kamDI_metrics = Metrics(kamDI_actual, kamDI_predicted, kamDI_protected)
    metrics[2].append(ret_res(kamDI_metrics))
    time[2].append(kamDI_time)

  collected_metrics = [[],[],[]]
  for x in range(len(metrics)):
    for i in range(5):
      m = []
      for j in range(len(metrics[x])):
        m.append(metrics[x][j][i])
      collected_metrics[x].append(m)

  for i in range(0,len(time)):
    if 'NA' in time[i]:
      time[i] = [k for k in time[i] if k != 'NA']
    if len(time[i]) == 0:
      final_time.append('0:00:00.00')
    else:
      mean_time = sum(time[i], timedelta()) / len(time[i])
      final_time.append(str(mean_time))

  for i in range(len(collected_metrics)):
    for x in range(len(collected_metrics[i])):
      if 'NA' in collected_metrics[i][x]:
        collected_metrics[i][x] = [k for k in collected_metrics[i][x] if k != 'NA']
      if len(collected_metrics[i][x]) == 0:
        final_metrics[i].append('NA')
        sd[i].append('NA')
      else:
        mean = sum(collected_metrics[i][x][0:len(collected_metrics[i][x])])/len(collected_metrics[i][x])
        sd[i].append(get_sd(collected_metrics[i][x], mean))
        final_metrics[i].append(mean)

  for i in range(len(final_metrics)):
    final_metrics[i].append(final_time[i])
    final_metrics[i].insert(1,sd[i][0])
    final_metrics[i].insert(3,sd[i][1])
    final_metrics[i].insert(5,sd[i][2])
    final_metrics[i].insert(7,sd[i][3])
    final_metrics[i].insert(9,sd[i][4])

  final_metrics[0].insert(0,'Kamishima Acc/DI')
  final_metrics[1].insert(0,'Kamishima Acc')
  final_metrics[2].insert(0,'Kamishima DI')

  kamboth = ','.join(map(str,final_metrics[0]))
  kamacc = ','.join(map(str,final_metrics[1]))
  kamDI = ','.join(map(str,final_metrics[2])) 

  return kamboth, kamacc, kamDI 

def run_zafar(data, times):
  metrics = [[],[],[],[]]
  time = [[],[],[],[]]
  sd = [[],[],[],[]]
  final_time = []
  final_metrics = [[],[],[],[]]

  for i in range(times):
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

    params["sep_constraint"] = 1
    params["gamma"] = 1000.0
    algorithm = ZafarAlgorithm(data, params)
    zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected, zafar_nopos_classification_time = algorithm.run()

    zafar_unconstrained_metrics = Metrics(zafar_unconstrained_actual, zafar_unconstrained_predicted, zafar_unconstrained_protected)
    metrics[0].append(ret_res(zafar_unconstrained_metrics))
    time[0].append(zafar_unconstrained_time)
    zafar_opt_accuracy_metrics = Metrics(zafar_opt_accuracy_actual, zafar_opt_accuracy_predicted, zafar_opt_accuracy_protected)
    metrics[1].append(ret_res(zafar_opt_accuracy_metrics))
    time[1].append(zafar_opt_accuracy_time)
    zafar_opt_fairness_metrics = Metrics(zafar_opt_fairness_actual, zafar_opt_fairness_predicted, zafar_opt_fairness_protected)
    metrics[2].append(ret_res(zafar_opt_fairness_metrics))
    time[2].append(zafar_opt_fairness_time)
    zafar_nopos_classification_metrics = Metrics(zafar_nopos_classification_actual, zafar_nopos_classification_predicted, zafar_nopos_classification_protected)
    metrics[3].append(ret_res(zafar_nopos_classification_metrics))
    time[3].append(zafar_nopos_classification_time)

  collected_metrics = [[],[],[],[]]
  for x in range(len(metrics)):
    for i in range(5):
      m = []
      for j in range(len(metrics[x])):
        m.append(metrics[x][j][i])
      collected_metrics[x].append(m)

  for i in range(0,len(time)):
    if 'NA' in time[i]:
      time[i] = [k for k in time[i] if k != 'NA']
    if len(time[i]) == 0:
      final_time.append('0:00:00.00')
    else:
      mean_time = sum(time[i], timedelta()) / len(time[i])
      final_time.append(str(mean_time))

  for i in range(len(collected_metrics)):
    # x is list of lists where each list is a metric 
    for x in range(len(collected_metrics[i])):
      if 'NA' in collected_metrics[i][x]:
        collected_metrics[i][x] = [k for k in collected_metrics[i][x] if k != 'NA']
      if len(collected_metrics[i][x]) == 0:
        final_metrics[i].append('NA')
        sd[i].append('NA')
      else:
        mean = sum(collected_metrics[i][x][0:len(collected_metrics[i][x])])/len(collected_metrics[i][x])
        sd[i].append(get_sd(collected_metrics[i][x], mean))
        final_metrics[i].append(mean)

  for i in range(len(final_metrics)):
    final_metrics[i].append(final_time[i])
    final_metrics[i].insert(1,sd[i][0])
    final_metrics[i].insert(3,sd[i][1])
    final_metrics[i].insert(5,sd[i][2])
    final_metrics[i].insert(7,sd[i][3])
    final_metrics[i].insert(9,sd[i][4])

  final_metrics[0].insert(0,'Zafar Unconstrained')
  final_metrics[1].insert(0,'Zafar w Acc Constraint')
  final_metrics[2].insert(0,'Zafar w Fair Constraint')
  final_metrics[3].insert(0,'Zafar No Pos Misclass')

  zafar_unconstrained = ','.join(map(str,final_metrics[0]))
  zafar_opt_accuracy = ','.join(map(str,final_metrics[1]))
  zafar_opt_fairness = ','.join(map(str,final_metrics[2]))
  zafar_nopos_classification = ','.join(map(str,final_metrics[3]))

  return zafar_unconstrained, zafar_opt_accuracy, zafar_opt_fairness, zafar_nopos_classification 

def run_metrics(data, times=10):
  with open("results/"+data+"-gen.csv",'w') as f:
    f.write('Algorithms,Acc,Acc_SD,BCR,BCR_SD,MCC,MCC_SD,DI,DI_SD,CV,CV_SD,Run Time'+'\n')

    # GEN
    svm, nb, lr = run_gen(data,times)
    f.write(svm+'\n')
    f.write(nb+'\n')
    f.write(lr+'\n')
  f.close()

  with open("results/"+data+"-calders.csv",'w') as f:
    f.write('Algorithms,Acc,Acc_SD,BCR,BCR_SD,MCC,MCC_SD,DI,DI_SD,CV,CV_SD,Run Time'+'\n')

    # CALDERS
    c2nb = run_calders(data,times)
    f.write(c2nb+'\n')
  f.close()

  with open("results/"+data+"-feldman.csv",'w') as f:
    f.write('Algorithms,Acc,Acc_SD,BCR,BCR_SD,MCC,MCC_SD,DI,DI_SD,CV,CV_SD,Run Time'+'\n')

    '''
    # FELDMAN
    feldman_svm, feldman_dt = run_feldman(data,times)
    f.write(feldman_svm+'\n')
    f.write(feldman_dt+'\n') 
  f.close()

  with open("results/"+data+"-kamishima.csv",'w') as f:
    f.write('Algorithms,Acc,Acc_SD,BCR,BCR_SD,MCC,MCC_SD,DI,DI_SD,CV,CV_SD,Run Time'+'\n')

    # KAMISHIMA
    kamboth, kamacc, kamDI = run_kamishima(data,times)
    f.write(kamboth+'\n')
    f.write(kamacc+'\n')
    f.write(kamDI+'\n')
  f.close()
  '''

  with open("results/"+data+"-zafar.csv",'w') as f:
    f.write('Algorithms,Acc,Acc_SD,BCR,BCR_SD,MCC,MCC_SD,DI,DI_SD,CV,CV_SD,Run Time'+'\n')

    # ZAFAR
    zafar_unconstrained, zafar_opt_accuracy, zafar_opt_fairness, zafar_nopos_classification = run_zafar(data,times)
    f.write(zafar_unconstrained+'\n')
    f.write(zafar_opt_accuracy+'\n')
    f.write(zafar_opt_fairness+'\n')
    f.write(zafar_nopos_classification)
  f.close()

if __name__ == '__main__':
  '''
  print("Analyzing German data...")
  run_metrics('german')
  print("Complete.")

  print('Analyzing Ricci data...')
  run_metrics("ricci")
  print('Complete.')
  print("\n")

  print('Analyzing Adult data...')
  run_metrics('adult')
  print('Complete.')
  print("\n")

  '''
  print('Analyzing Retailer data...')
  run_metrics("retailer")
  print('Complete.')
