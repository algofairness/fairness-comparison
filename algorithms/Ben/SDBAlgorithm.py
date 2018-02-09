#!/usr/bin/env python3

import random
from algorithms.Ben import boosting
from algorithms.Ben import svm
from algorithms.Ben import lr
#import lr
from data.objects import Adult, German
from algorithms.Ben.weaklearners.decisionstump import buildDecisionStump
#from algorithms.Ben.errorfunctions import signedStatisticalParity, labelError, individualFairness
from algorithms.Ben import errorfunctions
from algorithms.Ben.utils import arrayErrorBars, errorBars, experimentCrossValidate
from algorithms.Ben import utils
#from margin import svmRBFMarginAnalyzer, svmLinearMarginAnalyzer, boostingMarginAnalyzer, lrSKLMarginAnalyzer
from algorithms.Ben.margin import *
from algorithms.Algorithm import Algorithm

class SDBAlgorithm(Algorithm):
   def __init__(self, algorithm):
      Algorithm.__init__(self)
      self.model = algorithm
      self.name = 'SDB-' + self.model.get_name()

   def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

   @arrayErrorBars(2)
   def statistics(self, train, test, protectedIndex, protectedValue, learner):
      h = learner(train, protectedIndex, protectedValue)
     # print("Computing error")
      error = labelError(test, h)
     # print("Computing bias")
      bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
     # print("Computing UBIF")
      ubif = individualFairness(train, learner, 0.2, passProtected=True)
      return error, bias, ubif

   def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
      train = train_df.values.tolist()
      train_labels=[]
      train_data_points=[]
      test_labels=[]
      test_data_points=[]
      test = test_df.values.tolist()
      #print("train------------------",len(train),train)
      for datapoints in train:
         #print("in run------ datapoints",datapoints[-1])
         if (datapoints[-1] == 0):
            datapoints[-1] = -1
         #p=set(datapoints[:-1])
         #l=set(datapoints[-1])
         #dp=set((p,l))
         train_labels.append(int(datapoints[-1]))
         #datapoints[:-1]=set(datapoints[:-1])
         train_data_points.append(tuple(datapoints[:-1]))
      train= list(zip(train_data_points,train_labels))
      #print ("in run-----------------", train[:7])
      for datapoints in test:
         if (datapoints[-1] == 0.0):
            datapoints[-1] = -1.0
         #p=set(datapoints[:-1])
         #l=set(datapoints[-1])
         #dp=set((p,l))
         test_labels.append(int(datapoints[-1]))
         test_data_points.append(tuple(datapoints[:-1]))

      test= list(zip(test_data_points,test_labels))
         #train_set=set.add(dp)
      #print ("datapoints------------------",datapoint[0])
         #train=s.add(datapoints)

      #print("TRain_df------------------------------",train)
      protectedIndex = train_df.columns.get_loc(sensitive_attrs[0])
      protectedValue = privileged_vals[0]
      #print("dataset---------------",class_attr)
      #learner = str(self.model)+'Learner'
      #experimentCrossValidate(train, test, learner, 5, self.statistics, protectedIndex, protectedValue)
      return self.runAll(train,test, protectedIndex, protectedValue)
      #return self.model(train, protectedIndex, protectedValue)

   def lrLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = lrSKLMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def boostingLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def svmLearner(self, train, protectedIndex, protectedValue):
     # print("in svm learner--------")
      marginAnalyzer = svmRBFMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def svmLinearLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = svmLinearMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   #
   


   #@errorBars(10)
   def indFairnessStats(self, train, learner):
      #print("Computing UBIF")
      ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
      print("UBIF:", ubif)
      return ubif

   def runAll(self, test, train, protectedIndex, protectedValue):
      print("Shifted Decision Boundary Relabeling")
      dataset = test+train
      experiments = [
      ('SVM', self.svmLearner),
      ('SVMlinear', self.svmLinearLearner),
      ('AdaBoost', self.boostingLearner),
      #('LR', self.lrLearner)
         ]

      for (learnerName, learner) in experiments:
         print("%s" % (learnerName), flush=True)
         experimentCrossValidate(train,test, learner, 3, self.statistics, protectedIndex, protectedValue)
