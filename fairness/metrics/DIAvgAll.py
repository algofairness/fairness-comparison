import math

from fairness.metrics.utils import calc_prob_class_given_sensitive
from fairness.metrics.Metric import Metric

class DIAvgAll(Metric):
    """
    This metric calculates disparate imapct in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    If there are no positive protected classifications, 0.0 is returned.

    If there are multiple protected classes, the average DI over all groups is returned.
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'DIavgall'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        sensitive_values = list(set(sensitive))

        if len(sensitive_values) <= 1:
             print("ERROR: Attempted to calculate DI without enough sensitive values:" + \
                   str(sensitive_values))
             return 1.0

        # this list should only have one item in it
        single_unprotected = [val for val in sensitive_values if val in unprotected_vals][0]
        unprotected_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred,
                                                           single_unprotected)
        sensitive_values.remove(single_unprotected)
        total = 0.0
        for sens in sensitive_values:
             pos_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred, sens)
             DI = 0.0
             if unprotected_prob > 0:
                 DI = pos_prob / unprotected_prob
             if unprotected_prob == 0.0 and pos_prob == 0.0:
                 DI = 1.0
             total += DI

        if total == 0.0:
             return 1.0

        return total / len(sensitive_values)

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2
