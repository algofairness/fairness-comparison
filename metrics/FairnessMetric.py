from metrics.Metric import Metric

class FairnessMetric(Metric):
    def __init__(self):
        Metric.__init__(self)

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
        """
        Inputs: the actual results on the test set, the predicted results, a vector of the associated
        sensitive values, a list of the unprotected values for the sensitive categories, and 
        the positive value of the prediction task.  The actual and predicted results and the 
        sensitive attributes vector should have the same length (the length of the test set).
        """
        raise NotImplementedError("calc() in FairnessMetric is not implemented")
