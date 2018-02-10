#!/usr/bin/env python3

import random
from algorithms.Ben import boosting
from algorithms.Ben import svm
from algorithms.Ben import lr

from data.objects import Adult, German
from algorithms.Ben.weaklearners.decisionstump import buildDecisionStump
from algorithms.Ben import errorfunctions
from algorithms.Ben.utils import arrayErrorBars, errorBars, experimentCrossValidate
from algorithms.Ben import utils
from algorithms.Ben.margin import *
from algorithms.Algorithm import Algorithm

class SDBSVM(Algorithm):
   def __init__(self):
        Algorithm.__init__(self)
        self.name = "SDBSVM"

   def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

   @arrayErrorBars(2)
   def statistics(self, train, test, protectedIndex, protectedValue, learner):
      #print("in statistics-----------------------")
      h = learner(train, protectedIndex, protectedValue)
      print("Computing error")
      error = labelError(test, h)
      print("Computing bias")
      bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
      print("Computing UBIF")
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
      for datapoints in train:
         if (datapoints[-1] == 0):
            datapoints[-1] = -1
         train_labels.append(int(datapoints[-1]))
         train_data_points.append(tuple(datapoints[:-1]))
      train= list(zip(train_data_points,train_labels))
      for datapoints in test:
         if (datapoints[-1] == 0.0):
            datapoints[-1] = -1.0
         test_labels.append(int(datapoints[-1]))
         test_data_points.append(tuple(datapoints[:-1]))

      test= list(zip(test_data_points,test_labels))
      protectedIndex = train_df.columns.get_loc(sensitive_attrs[0])
      protectedValue = privileged_vals[0]

      return self.runAll(train,test, protectedIndex, protectedValue) 
   

   def svmLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = svmRBFMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def svmLinearLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = svmLinearMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)

   @errorBars(10)
   def indFairnessStats(self, train, learner):
      print("Computing UBIF")
      ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
      print("UBIF:", ubif)
      return ubif

   def runAll(self, test, train, protectedIndex, protectedValue):
      print("Shifted Decision Boundary Relabeling")
      dataset = test+train
      experiments = [
      ('SVM', self.svmLearner),
      ('SVMlinear', self.svmLinearLearner)
         ]

      for (learnerName, learner) in experiments:
         print("%s" % (learnerName), flush=True)
         experimentCrossValidate(train,test, learner, 5, self.statistics, protectedIndex, protectedValue)
      