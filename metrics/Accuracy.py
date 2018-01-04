from metrics.Metric import Metric
from sklearn.metrics import accuracy_score

class Accuracy(Metric):
    def __init__(self, actual, predicted):
        Metric.__init__(self, actual, predicted)
        self.name = 'accuracy'

    def calc(self):
        return accuracy_score(self.actual, self.predicted)
