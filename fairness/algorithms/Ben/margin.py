from fairness.algorithms.Ben.utils import *
from fairness.algorithms.Ben.errorfunctions import *

from fairness.algorithms.Ben import svm
import numpy
from fairness.algorithms.Ben import lr
from fairness.algorithms.Ben import boosting
from fairness.algorithms.Ben.weaklearners.decisionstump import buildDecisionStump
import random

try:
   import matplotlib.pyplot as plt
except ImportError:
   pass


class marginAnalyzer(object):
   def __init__(self, data=None, defaultThreshold=None, marginRange=None,
                  protectedIndex=None, protectedValue=None, bulkMargin=None):
      self.defaultThreshold = defaultThreshold
      if data is not None and 'trainingData' not in dir(self):
         self.splitData(data)

      if bulkMargin is None:
         self.trainingMargins = [self.margin(x[0]) for x in self.trainingData]
         self.validationMargins = [self.margin(x[0]) for x in self.validationData]
         self.bulkMargin = None
      else:
         self.trainingMargins = bulkMargin([x[0] for x in self.trainingData])
         self.validationMargins = bulkMargin([x[0] for x in self.validationData])
         self.bulkMargin = bulkMargin

      self.margins = numpy.concatenate([self.trainingMargins, self.validationMargins], axis=0)
      self.margins=self.margins.ravel()
      
      self.minMargin, self.maxMargin = min(self.margins), max(self.margins)
      self.minShift, self.maxShift =  self.minMargin - self.defaultThreshold, self.maxMargin - self.defaultThreshold
      if marginRange != None:
         self.marginRange = marginRange
      else:
         self.marginRange = (self.minMargin, self.maxMargin)
      self.setProtected(protectedIndex, protectedValue)

   def margin(self, x):
      raise NotImplementedError()

   def splitData(self, data):
      self.data = random.sample(data,len(data))
      self.trainingData = data[:len(data)//2]
      self.validationData = data[len(data)//2:]

   def setProtected(self, protectedIndex, protectedValue):
      assert protectedIndex is not None
      assert protectedValue is not None
      self.protectedIndex = protectedIndex
      self.protectedValue = protectedValue

   #returns True if x is protected, False otherwise; can be used as a condition for conditionalShiftClassifier
   def protected(self, x):
      assert self.protectedIndex is not None
      assert self.protectedValue is not None
      return x[self.protectedIndex] == self.protectedValue


   #returns a classifier which takes a data point as an input and returns 1 if margin is above threshold, 0 otherwise
   def classifier(self, threshold=None):
      if threshold == None:
         threshold = lambda x: self.defaultThreshold
      return lambda x: 1  if  self.margin(x) >= threshold(x) else 0


   #returns a classifier with shifted threshold for data points satisfying condition
   def conditionalShiftClassifier(self, shift, condition=None):
      if condition == None:
         condition = self.protected
      return self.classifier(lambda x: self.defaultThreshold + shift if condition(x) else self.defaultThreshold)


   def conditionalMarginShiftedLabels(self, data, margins, shift, condition):
      # condition is margin >= threshold + shift, so that if shift is negative
      # the threshold is lower.
      shiftedMargins = [(m-shift if condition(x[0]) else m) for (m, x) in zip(margins, data)]
      labels = [1 if m >= self.defaultThreshold else 0 for m in shiftedMargins]
      return labels


   #finds the shift which achieves goal=0 under condition
   #goal takes two arguments, data and h
   def optimalShift(self, goal=None, condition=None, rounds=3):
      #print("in optimalshift function--------------------")
      if goal == None:
         goal = lambda d, h: signedStatisticalParity(d, self.protectedIndex, self.protectedValue, h)
      if condition == None:
         condition = self.protected

      low = self.minShift
      high = self.maxShift
      dataToUse = self.validationData

      minGoalValue = goal(dataToUse, self.conditionalShiftClassifier(low, condition))
      maxGoalValue = goal(dataToUse, self.conditionalShiftClassifier(high, condition))
     

      if sign(minGoalValue) != sign(maxGoalValue):
         # a binary search for zero
         for _ in range(rounds):
            midpoint = (low + high) / 2
            if (sign(goal(dataToUse, self.conditionalShiftClassifier(low, condition))) ==
                  sign(goal(dataToUse, self.conditionalShiftClassifier(midpoint, condition)))):
               low = midpoint
            else:
               high = midpoint
         return midpoint
      else:
         print("Warning: bisection method not applicable")
         bestShift = None
         bestVal = float('inf')
         step = (high-low)/rounds
         for newShift in numpy.arange(low, high, step):
            newVal = goal(dataToUse, self.conditionalShiftClassifier(newShift, condition))
            #print(newVal)
            newVal = abs(newVal)
            if newVal < bestVal:
               bestShift = newShift
               bestVal = newVal
         return bestShift

   def optimalShiftClassifier(self, goal=None, condition=None, rounds=3):
      if goal == None:
         goal = lambda d, h: signedStatisticalParity(d, self.protectedIndex, self.protectedValue, h)
      if condition == None:
         condition = self.protected
      return self.conditionalShiftClassifier(self.optimalShift(goal, condition, rounds), condition)


class boostingMarginAnalyzer(marginAnalyzer):
   def __init__(self, data, protectedIndex, protectedValue, numRounds=3,
               weakLearner=buildDecisionStump, computeError=boosting.weightedLabelError):

      self.splitData(data)
      _, self.hypotheses, self.alphas = boosting.detailedBoost(self.trainingData, numRounds, weakLearner, computeError)
      super().__init__(defaultThreshold=0, marginRange=(-1,1), protectedIndex=protectedIndex,
                  protectedValue=protectedValue)



   def margin(self, x):
      return boosting.margin(x, self.hypotheses, self.alphas)


class svmRBFMarginAnalyzer(marginAnalyzer):
   
   def __init__(self, data, protectedIndex, protectedValue, gamma=svm.DEFAULT_GAMMA):
      self.splitData(data)
      outputs = svm.svmDetailedSKL(self.trainingData, gamma, verbose=True, kernel='rbf')
      self.svmDetails = outputs
      self.bulkMargin = outputs[-2]
      self.margin = outputs[-1]
      super().__init__(defaultThreshold=0, protectedIndex=protectedIndex,
                     protectedValue=protectedValue, bulkMargin=self.bulkMargin)

class svmLinearMarginAnalyzer(marginAnalyzer):
   def __init__(self, data, protectedIndex, protectedValue, gamma=svm.DEFAULT_GAMMA):
      self.splitData(data)
      outputs = svm.svmDetailedSKL(self.trainingData, gamma, verbose=True, kernel='linear')
      self.svmDetails = outputs
      self.bulkMargin = outputs[-2]
      self.margin = outputs[-1]
      super().__init__(defaultThreshold=0, protectedIndex=protectedIndex,
                     protectedValue=protectedValue, bulkMargin=self.bulkMargin)

class lrSKLMarginAnalyzer(marginAnalyzer):
   def __init__(self, data, protectedIndex, protectedValue):
      self.splitData(data)
      self.margin = lr.lrDetailedSKL(self.trainingData)[0]
      super().__init__(defaultThreshold=0.5, marginRange=(0,1), protectedIndex=protectedIndex,
                     protectedValue=protectedValue)





