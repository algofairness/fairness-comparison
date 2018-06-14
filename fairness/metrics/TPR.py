from fairness.metrics.Metric import Metric
from sklearn.metrics import recall_score

class TPR(Metric):
    """
    Returns the true positive rate (aka recall) for the predictions.  Assumes binary
    classification.
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'TPR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return recall_score(actual, predicted, pos_label=positive_pred, average='binary')

