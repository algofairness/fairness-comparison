from fairness.metrics.Metric import Metric
from fairness.metrics.TNR import TNR
from fairness.metrics.TPR import TPR

class BCR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'BCR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tnr = TNR()
        tnr_val = tnr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        tpr = TPR()
        tpr_val = tpr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        bcr = (tpr_val + tnr_val) / 2.0
        return bcr
