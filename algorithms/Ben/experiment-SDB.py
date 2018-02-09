#!/usr/bin/env python3

import random
import boosting
import svm
import lr
from data import adult, german, singles
from weaklearners.decisionstump import buildDecisionStump
from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import errorBars, arrayErrorBars, sign, variance, experimentCrossValidate
from margin import svmRBFMarginAnalyzer, svmLinearMarginAnalyzer, boostingMarginAnalyzer, lrSKLMarginAnalyzer
from algorithms.Algorithm import Algorithm

class SDBAlgorithm(Algorithm):
   def __init__(self, algorithm):
      Algorithm.__init__(self)
      self.model = algorithm
      self.name = 'SDB-' + self.model.get_name()

   def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
      train = train_df.values.tolist()
      test = test_df.values.tolist()
      protectedIndex = train_df.columns.get_loc(sensitive_attrs[0])
      protectedValue = privileged_vals
      return self.model(train, protectedIndex, protectedValue)

   def lrLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = lrLearnerMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def boostingLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


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


   @arrayErrorBars(2)
   def statistics(self, train, test, protectedIndex, protectedValue, learner):
      h = learner(train, protectedIndex, protectedValue)
      print("Computing error")
      error = labelError(test, h)
      print("Computing bias")
      bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
      print("Computing UBIF")
      ubif = individualFairness(train, learner, 0.2, passProtected=True)
      return error, bias, ubif


   @errorBars(10)
   def indFairnessStats(self, train, learner):
      print("Computing UBIF")
      ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
      print("UBIF:", ubif)
      return ubif

   """def runAll():
      print("Shifted Decision Boundary Relabeling")
      experiments = [
         (('SVM', svmLearner), adult),
         (('SVMlinear', svmLinearLearner), german),
         (('SVM', svmLearner), singles),
         (('AdaBoost', boostingLearner), adult),
         (('AdaBoost', boostingLearner), german),
         (('AdaBoost', boostingLearner), singles),
         (('LR', lrLearner), adult),
         (('LR', lrLearner), german),
         (('LR', lrLearner), singles),
      ]

      for (learnerName, learner), dataset in experiments:
         print("%s %s" % (dataset.name, learnerName), flush=True)
         experimentCrossValidate(dataset, learner, 5, statistics)


   if __name__ == '__main__':
     runAll() """
