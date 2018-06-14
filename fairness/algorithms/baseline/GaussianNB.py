from sklearn.naive_bayes import GaussianNB as SKLearn_NB
from fairness.algorithms.baseline.Generic import Generic

class GaussianNB(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_NB()
        self.name = "GaussianNB"
