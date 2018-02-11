import random
import math
import numpy

# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
   choice = random.uniform(0, sum(weights))
   choiceIndex = 0

   for weight in weights:
      choice -= weight
      if choice <= 0:
         return choiceIndex

      choiceIndex += 1


# normalize a distribution
def normalize(weights):
   norm = sum(weights)
   return tuple(m / norm for m in weights)


def sign(x):
   return 1 if x >= 0 else 0


def zeroOneSign(x):
	return 1 if x >= 0 else 0


def median(L):
    L.sort()
    half = int(len(L) / 2)
    if len(L) % 2 == 0:
        return (L[half+1] + L[half]) / 2.0
    else:
        return L[half]


def avg(L):
   return sum(L) / len(L)


def variance(L):
   average = avg(L)
   squaredDeviations = [(x - average)**2 for x in L]
   return (1 / (len(L) - 1)) * sum(squaredDeviations)


def column(A, j):
   return [row[j] for row in A]


def transpose(A):
   return [column(A, j) for j in range(len(A[0]))]


# take any function f which produces a number and produce a function which
# outputs aggregate statistics from calling f n times.
def errorBars(n):
   def errorbarDecorator(f):
      def newF(*args):
         results = [f(*args) for _ in range(n)]
         return avg(results), min(results), max(results), variance(results)
      return newF
   return errorbarDecorator


# compute coordinatewise error bars for an array-valued function
def arrayErrorBars(n):
   def errorbarDecorator(f):
      def newF(*args):
         results = [f(*args) for _ in range(n)]
         return [(avg(x), min(x), max(x), variance(x)) for x in transpose(results)]
      return newF
   return errorbarDecorator


# compute the min and return the argument providing the min
def argmin(L):
   if len(L) == 0:
      raise ValueError("Empty list")

   theMin = L[0]
   minIndex = 0

   for i,x in enumerate(L, start=1):
      if x < theMin:
         minIndex = i
         theMin = x

   return minIndex, theMin

#normalize a matrix such that each entry is between 0 and 1
def normalize01(data):
   a = [min(row[j] for row in data) for j in range(len(data[0]))]
   b = [max(row[j] for row in data) for j in range(len(data[0]))]
   return [tuple([(row[j]-a[j])/(b[j]-a[j]) if b[j]!=a[j] else 0 for j in range(len(row))]) for row in data]

def lpNorm(v, p):
   return math.pow(sum(math.pow(abs(x), p) for x in v), 1/p)

def lpDistance(u, v, p):
   assert len(u)==len(v)
   return lpNorm([u[i]-v[i] for i in range(len(u))], p)


def sigmoid(z):
   return 1.0 / (1 + numpy.exp(-z))
 
def experimentCrossValidate(Train, Test, learner, times, statistics, protectedIndex, protectedValue, massage=False):
   PI = protectedIndex
   PV = protectedValue
   originalTrain = Train
   originalTest = Test
   allData = Train+Test
   #allData= allData.values.tolist()
   variances = [[], [], []] #error, bias, ubif
   mins = [float('inf'), float('inf'), float('inf')]
   maxes = [-float('inf'), -float('inf'), -float('inf')]
   avgs = [0, 0, 0]
   
   for time in range(times):
     random.shuffle(allData)
     train = allData[:len(originalTrain)]
     test = allData[len(originalTrain):]
     if not massage:
       classifier_t = learner(train, protectedIndex, protectedValue)
       output = statistics(train, test, PI, PV, learner)
     else:
       from massaging import randomOneSideMassageData
       classifier_t = learner(train, protectedIndex, protectedValue)
       output = statistics(randomOneSideMassageData, train, test, PI, PV, learner)
     
     for i in range(len(output)):
       avgs[i] += (output[i][0] - avgs[i]) / (time + 1)
       mins[i] = min(mins[i], output[i][1])
       maxes[i] = max(maxes[i], output[i][2])
       variances[i].append(output[i][0]) # was too lazy to implement online alg
       # warning: this doesn't take into account the variance of each split
   
   for i in range(len(variances)):
     variances[i] = variance(variances[i])
  #prediction on test data
   prediction=[]
   for datapoints in Test:
    y=classifier_t(datapoints[0])
    prediction.append(y)
   return prediction
