from metrics.Metric import Metric
from sklearn.metrics import accuracy_score

class Accuracy(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'accuracy'

    def calc(self, actual, predicted):
        return accuracy_score(actual, predicted)
