import math
import numpy
from numpy.linalg import norm
import random
from fairness.algorithms.Ben import utils
#from utils import sign

DEFAULT_NUM_ROUNDS = 1
DEFAULT_LAMBDA = 1.0
DEFAULT_GAMMA = 0.1


def hyperplaneToHypothesis(w):
   return lambda x: sign(numpy.dot(w,x))


# use scikit-learn to do the svm for us
def svmDetailedSKL(data, gamma=DEFAULT_GAMMA, verbose=False, kernel='rbf'):
  # if verbose:
   #  print("Loading scikit-learn")
   from sklearn import svm
   points, labels = zip(*data)
   clf = svm.SVC(kernel=kernel, gamma=gamma)

   #if verbose:
   #   print("Training classifier")

   skClassifier = clf.fit(points, labels)
   hypothesis = lambda x: skClassifier.predict([x])[0]
   bulkHypothesis = lambda data: skClassifier.predict(data)

   alphas = skClassifier.dual_coef_[0]
   supportVectors = skClassifier.support_vectors_
   error = lambda data: 1 - skClassifier.score(*zip(*data))

   intercept = skClassifier.intercept_
   margin = lambda y: skClassifier.decision_function([y])[0]
   bulkMargin = lambda pts: skClassifier.decision_function(pts)

   #if verbose:
   #   print("Done")

   return (hypothesis, bulkHypothesis, skClassifier, error, alphas, intercept,
            gamma, supportVectors, bulkMargin, margin)


def svmSKL(data, gamma=DEFAULT_GAMMA, verbose=False, kernel='rbf'):
   return svmDetailedSKL(data, gamma, verbose, kernel)[0]

def svmLinearSKL(data, verbose=False):
   return svmDetailedSKL(data, 0, verbose, 'linear')[0]

# compute the margin of a point
def margin(point, hyperplane):
   return numpy.dot(hyperplane, point)

# compute the absolute value of the margin of a point
def absMargin(point, hyperplane):
   return abs(margin(point, hyperplane))
