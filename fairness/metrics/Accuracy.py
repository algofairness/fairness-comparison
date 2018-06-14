from fairness.metrics.Metric import Metric
from sklearn.metrics import accuracy_score

class Accuracy(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'accuracy'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return accuracy_score(actual, predicted)
