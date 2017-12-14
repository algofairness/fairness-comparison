from metrics.Metric import Metric

class Accuracy(Metric):
    def __init__(self, actual, predicted):
        Metric.__init__(actual, predicted)

    def calc(self):
        return accuracy_score(self.actual, self.predicted)
