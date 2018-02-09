from algorithms.Ben import utils 
import random
import heapq


def minLabelErrorOfHypothesisAndNegation(data, h):
   posData, negData = ([(x, y) for (x, y) in data if h(x) == 1],
                       [(x, y) for (x, y) in data if h(x) == -1])

   posError = sum(y == -1 for (x, y) in posData) + sum(y == 1 for (x, y) in negData)
   negError = sum(y == 1 for (x, y) in posData) + sum(y == -1 for (x, y) in negData)
   return min(posError, negError) / len(data)


def makeLinearCombination(error1, error2, error1Weight):
   def customError(data, h):
      e1 = error1(data, h)
      e2 = error2(data, h)
      return error1Weight * e1 + (1 - error1Weight) * e2

   return customError


def precomputedLabelError(data, labels):
   return sum(1 for (x,l) in zip(data, labels) if x[1] != l) / len(data)


def labelError(data, h):
   return len([1 for (x,y) in data if h(x) != y]) / len(data)


# data is a list of unlabeled examples
def precomputedLabelStatisticalParity(data, labels, protectedIndex, protectedValue, weights=None):
   if weights == None:
      weights = [1]*len(data)
      #print("----------------------PI--------------PV------------------------",protectedIndex, float(protectedValue))
      #print("data in errorfunction------------------------------------",data)
   protectedClass = [(x,wt,l) for (x,wt,l) in zip(data, weights, labels)
                        if x[protectedIndex] == protectedValue]
   elseClass  = [(x,wt,l) for (x,wt,l) in zip(data, weights, labels)
                        if x[protectedIndex] != protectedValue]

   if len(protectedClass) == 0:
      print("Nobody in the protected class")
      return sum(w for (x,w,l) in elseClass  if l == 1) / sum(w for (x,w,l) in elseClass)
   elif len(elseClass) == 0:
      print("Nobody in the else class")
      return -sum(w for (x,w,l) in protectedClass if l == 1) / sum(w for (x,w,l) in protectedClass)
   else:
      protectedProb = sum(w for (x,w,l) in protectedClass if l == 1) / sum(w for (x,w,l) in protectedClass)
      elseProb =  sum(w for (x,w,l) in elseClass  if l == 1) / sum(w for (x,w,l) in elseClass)

   return elseProb - protectedProb


# data is a list of labeled examples (this is weird; should be consistent)
def signedStatisticalParity(data, protectedIndex, protectedValue, h=None, weights=None):
   #print("in SP--------------------------",type(data[0]),len(data[0]))
   if len(data[0]) == 2: # should do better type checking here...
      pts, labels = zip(*data)
      #print("data----------",data)
      #print("labels--------",labels)
   else:
      pts = data
      labels = None

   if h is not None:
      #print("-------------------------if h is not none----------------------------",pts)
      labels = [h(x) for x in pts]
      #print("-------------------------if h is not none----------------------------",labels)

   if labels is None:
      raise Exception("Must provide either labels or a hypothesis to signedStatisticalParity")
   return precomputedLabelStatisticalParity(pts, labels, protectedIndex, protectedValue, weights)


def statisticalParity(data, protectedIndex, protectedValue, h=None, weights=None):
   return abs(signedStatisticalParity(data, protectedIndex, protectedValue, h, weights))


# add a new unbiased feature, introduce uniform random discrimination on that feature
# run the learner on the biased data, and then see how many of the flipped labels
# it can recover.
# learner: data -> classifier function
def individualFairness(data, learner, flipProportion=0.2, passProtected=False):
   protectedIndex = 0
   protectedValue = 0
   data=tuple(data)
   #print("data------------------------------------------------------",type(data), data[0])
   """for item in data:
      item=list(item)
      item[0]=tuple(item[0])
      #item=tuple(item)
      #list(item[0])=tuple(item[0])"""
   #print("data-----------------------------", data)
   unbiasedData = [((random.choice([0,1]),) + x[0], x[1]) for x in data]

   indicesOfProtected = [i for i,x in enumerate(unbiasedData)
                           if x[0][protectedIndex] == 0 and x[1] == 1]
   m = len(indicesOfProtected)

   indicesOfFlippedData = set(random.sample(indicesOfProtected, int(flipProportion * m)))
   biasedData = [(x[0], (-1 if i in indicesOfFlippedData else x[1])) for i,x in enumerate(unbiasedData)]

   if passProtected:
      h = learner(biasedData, protectedIndex, protectedValue)
   else:
      h = learner(biasedData)

   flippedPts = [x for i,x in enumerate(biasedData) if i in indicesOfFlippedData]
   error = sum(h(x) != y for (x,y) in flippedPts) / len(flippedPts)
   return error
