from metrics.Metric import Metric
import sys
import numpy

class CV(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'CV'

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
        predicted = numpy.array(predicted)
        sensitive = numpy.array(sensitive)
        
        unprotected_subset = numpy.full(sensitive.shape, False)
        for unprotected_val in unprotected_vals:
            a = (sensitive == unprotected_val)
            unprotected_subset = numpy.logical_or(unprotected_subset, a)
        protected_subset = numpy.logical_not(unprotected_subset)

        predicted_protected   = predicted[protected_subset]
        predicted_unprotected = predicted[unprotected_subset]

        v1 = (predicted_unprotected == positive_pred).sum() / len(predicted_unprotected)
        v2 = (predicted_protected == positive_pred).sum() / len(predicted_protected)

        CV = v1 - v2
        return (1 - CV/2)
