#!/usr/bin/env python3

import random
from fairness.algorithms.Ben import boosting
from fairness.algorithms.Ben import svm
from fairness.algorithms.Ben import lr

from fairness.data.objects import Adult, German
from fairness.algorithms.Ben.weaklearners.decisionstump import buildDecisionStump
from fairness.algorithms.Ben import errorfunctions
from fairness.algorithms.Ben.utils import arrayErrorBars, errorBars, experimentCrossValidate
from fairness.algorithms.Ben import utils
from fairness.algorithms.Ben.margin import *
from fairness.algorithms.Algorithm import Algorithm

class SDBAdaBoost(Algorithm):
   def __init__(self):
        Algorithm.__init__(self)
        self.name = "SDB-AdaBoost"

   def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

   @arrayErrorBars(2)
   def statistics(self, train, test, protectedIndex, protectedValue, learner):
      #print("in statistics-----------------------",train[0])
      h = learner(train, protectedIndex, protectedValue)
      #print("Computing error")
      error = labelError(test, h)
      #print("Computing bias")
      bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
      #print("Computing UBIF")
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
         if (int(datapoints[-1]) != 1):
            datapoints[-1] = 0
         train_labels.append(int(datapoints[-1]))
         train_data_points.append(tuple(datapoints[:-1]))
      train= list(zip(train_data_points,train_labels))
      for datapoints in test:
         if (int(datapoints[-1]) != 1):
            datapoints[-1] = 0
         test_labels.append(int(datapoints[-1]))
         test_data_points.append(tuple(datapoints[:-1]))

      test= list(zip(test_data_points,test_labels))
      protectedIndex = train_df.columns.get_loc(sensitive_attrs[0])
      protectedValue = privileged_vals[0]
      #print("test data ---------------------------", test[:6],"---------------------train-----------------",train[:8] )

      prediction=self.runAll(train,test, protectedIndex, protectedValue)
      #print("prediction-----",len(prediction))
      return  prediction, []

   def boostingLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)

   @errorBars(10)
   def indFairnessStats(self, train, learner):
      #print("Computing UBIF")
      ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
      #print("UBIF:", ubif)
      return ubif

   def runAll(self,train, test, protectedIndex, protectedValue):
      print("Shifted Decision Boundary Relabeling")
      dataset = test+train
      return experimentCrossValidate(train,test, self.boostingLearner, 2, self.statistics, protectedIndex, protectedValue)
      
