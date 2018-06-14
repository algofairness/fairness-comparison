from fairness.algorithms.Ben.errorfunctions import minLabelErrorOfHypothesisAndNegation
import sys


class Stump:
   def __init__(self):
      self.gtLabel = None
      self.ltLabel = None
      self.splitThreshold = None
      self.splitFeature = None

   def classify(self, point):
      if point[self.splitFeature] >= self.splitThreshold:
         return self.gtLabel
      else:
         return self.ltLabel

   def __call__(self, point):
      return self.classify(point)


def majorityVote(data):
   ''' Compute the majority of the class labels in the given data set. '''
   labels = [label for (pt, label) in data]
   try:
      return max(set(labels), key=labels.count)
   except:
      return -1


def bestThreshold(data, index, errorFunction):
   '''Compute best threshold for a given feature. Returns (threshold, error)
   Use errorFunction=negativeGain for maximum entropy gain'''

   thresholds = [point[index] for (point, label) in data]
   def makeThreshold(t):
      return lambda x: 1 if x[index] >= t else -1

   errors = [(threshold, errorFunction(data, makeThreshold(threshold))) for threshold in thresholds]
   return min(errors, key=lambda p: p[1])


def defaultError(data, h):
   return minLabelErrorOfHypothesisAndNegation(data, h)


def buildDecisionStump(drawExample, errorFunction=defaultError, debug=False, featureNames=None, forbiddenFeatures=()):
   # find the index of the best feature to split on, and the best threshold
   # for that index

   data = [drawExample() for _ in range(500)]

   bestThresholds = [(i,) + bestThreshold(data, i, errorFunction) for i in range(len(data[0][0])) if i not in forbiddenFeatures]
   feature, thresh, _ = min(bestThresholds, key = lambda p: p[2])

   stump = Stump()
   stump.splitFeature = feature
   stump.splitThreshold = thresh
   stump.gtLabel = majorityVote([x for x in data if x[0][feature] >= thresh])
   stump.ltLabel = majorityVote([x for x in data if x[0][feature] < thresh])

   if debug:
      if featureNames != None:
         sys.stderr.write('Feature: %s, threshold: %d, %s\n' % (featureNames[feature], thresh, '+' if stump.gtLabel == 1 else '-'))
      else:
         sys.stderr.write('Feature: %d, threshold: %d, %s\n' % (feature, thresh, '+' if stump.gtLabel == 1 else '-'))

   return stump
