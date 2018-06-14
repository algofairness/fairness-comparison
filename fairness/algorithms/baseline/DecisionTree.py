from sklearn.tree import DecisionTreeClassifier as SKLearn_DT
from fairness.algorithms.baseline.Generic import Generic

class DecisionTree(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_DT()
        self.name = "DecisionTree"
