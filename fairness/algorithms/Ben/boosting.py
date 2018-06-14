import math
from fairness.algorithms.Ben.utils import *
from fairness.algorithms.Ben.errorfunctions import labelError
from fairness.algorithms.Ben.weaklearners.decisionstump import buildDecisionStump

# compute the weighted error of a given hypothesis on a distribution
# return all of the hypothesis results and the error
def weightedLabelError(h, examples, weights):
   hypothesisResults = [h(x)*y for (x,y) in examples] # +1 if correct, else -1
   return hypothesisResults, sum(w for (z,w) in zip(hypothesisResults, weights) if z < 0)


# boost: [(list, label)], learner, int -> (list -> label)
# where a learner is (() -> (list, label)) -> (list -> label)
# boost the weak learner into a strong learner
def adaboostGenerator(examples, weakLearner, rounds, computeError=weightedLabelError):
   distr = normalize([1.0] * len(examples))
   hypotheses = [None] * rounds
   alpha = [0] * rounds

   for t in range(rounds):
     # print(t)
      def drawExample():
         return examples[draw(distr)]

      hypotheses[t] = weakLearner(drawExample)
      hypothesisResults, error = computeError(hypotheses[t], examples, distr)

      alpha[t] = 0.5 * math.log((1 - error) / (.0001 + error))
      distr = normalize([d * math.exp(-alpha[t] * r)
                         for (d,r) in zip(distr, hypothesisResults)])

      def weightedMajorityVote(x):
         return sign(sum(a * h(x) for (a, h) in zip(alpha, hypotheses[:t+1])))

      yield weightedMajorityVote, hypotheses[:t+1], alpha[:t+1]


#convenience wrapper for boosting
#returns the outputted hypothesis from boosting
def boost(trainingData, numRounds=20, weakLearner=buildDecisionStump, computeError=weightedLabelError):
   generator = adaboostGenerator(trainingData, weakLearner, numRounds, computeError)

   for h, _, _ in generator:
      pass

   return h


# call an optional diagnostic function to output round-wise intermediate results
# return more information at the end
def detailedBoost(trainingData, numRounds=20, weakLearner=buildDecisionStump, computeError=weightedLabelError, diagnostic=None):
   generator = adaboostGenerator(trainingData, weakLearner, numRounds, computeError)

   for h, hypotheses, alphas in generator:
      if diagnostic is not None:
         diagnostic({'h': h, 'hypoheses': hypotheses, 'alphas': alphas})

   return h, hypotheses, alphas


# compute the margin of a point with the label to express whether it's correct
# alpha is the weights of the hypotheses from the boosting algorithm
def marginWithLabel(point, label, hypotheses, alpha):
	return label * sum(a*h(point) for (h, a) in zip(hypotheses, alpha)) / sum(alpha)


# compute the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def margin(point, hypotheses, alpha):
	return sum(a*h(point) for (h, a) in zip(hypotheses, alpha)) / sum(alpha)


# compute the absolute value of the margin of a point
# alpha is the weights of the hypotheses from the boosting algorithm
def absMargin(point, hypotheses, alpha):
   return abs(margin(point, hypotheses, alpha))
