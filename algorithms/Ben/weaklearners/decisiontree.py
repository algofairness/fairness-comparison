import math

class Tree:
   def __init__(self, parent=None):
      self.parent = parent
      self.leftChild = None
      self.rightChild = None
      self.label = None
      self.classCounts = None
      self.splitThreshold = None
      self.splitFeature = None


def dataToDistribution(data):
   ''' Turn a dataset which has n possible classification labels into a
       probability distribution with n entries. '''
   allLabels = [label for (point, label) in data]
   numEntries = len(allLabels)
   possibleLabels = set(allLabels)

   return [float(allLabels.count(aLabel)) / numEntries for aLabel in possibleLabels]



def entropy(dist):
   ''' Compute the Shannon entropy of the given probability distribution. '''
   return -sum([p * math.log(p, 2) for p in dist])


def gain(data, index, threshold):
   entropyGain = entropy(dataToDistribution(data))

   dataSubsets = [
      [(point, label) for (point, label) in data if point[index] >= threshold],
      [(point, label) for (point, label) in data if point[index] < threshold]]

   for dataSubset in dataSubsets:
      entropyGain -= entropy(dataToDistribution(dataSubset))

   return entropyGain


def homogeneous(data):
   ''' Return True if the data have the same label, and False otherwise. '''
   return len(set([label for (point, label) in data])) <= 1



def majorityVote(data):
   ''' Compute the majority of the class labels in the given data set. '''
   labels = [label for (pt, label) in data]
   try:
      return max(set(labels), key=labels.count)
   except:
      return -1



def bestThreshold(data, index):
   thresholds = [point[index] for (point, label) in data]
   return max(thresholds, key=lambda t: gain(data, index, t))
 


def buildDecisionTree(data, root, remainingFeatures):
   ''' Build a decision tree from the given data, appending the children
       to the given root node (which may be the root of a subtree). '''

   if homogeneous(data):
      root.label = data[0][1]
      root.classCounts = {root.label: len(data)}
      return root

   if len(remainingFeatures) == 0:
      return majorityVote(data, root)

   bestThresholds = [(i, bestThreshold(data, i)) for i in remainingFeatures]
   feature, thresh = max(bestThresholds, key=lambda z: gain(data, z[0], z[1]))
   
   if gain(data, bestFeature) == 0:
      return majorityVote(data, root)

   root.splitFeature = bestFeature

   # add child nodes and process recursively
   for dataSubset in splitData(data, bestFeature):
      aChild = Tree(parent=root)
      aChild.splitFeatureValue = dataSubset[0][0][bestFeature]
      root.children.append(aChild)

      buildDecisionTree(dataSubset, aChild, remainingFeatures - set([bestFeature]))

   return root


def decisionTree(data):
   return buildDecisionTree(data, Tree(), set(range(len(data[0][0]))))


def classify(tree, point):
   ''' Classify a data point by traversing the given decision tree. '''

   if tree.children == []:
      return tree.label
   else:
      matchingChildren = [child for child in tree.children
         if child.splitFeatureValue >= point[tree.splitFeature]]

      if len(matchingChildren) == 0:
         raise Exception("Classify is not able to handle noisy data. Use classify2 instead.")

      return classify(matchingChildren[0], point)

