from sklearn.linear_model import LogisticRegression as SKLearn_LR
from fairness.algorithms.baseline.Generic import Generic

class LogisticRegression(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_LR()
        self.name = "LR"
