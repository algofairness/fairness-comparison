import math

from fairness.metrics.utils import calc_pos_protected_percents
from fairness.metrics.Metric import Metric

class DIBinary(Metric):
    """
    This metric calculates disparate imapct in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    If there are no positive protected classifications, 0.0 is returned.

    Multiple protected classes are treated as one large group, so that this compares the privileged
    class to all non-privileged classes as a group.
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'DIbinary'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_pos_percent, protected_pos_percent = \
            calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred)
        DI = 0.0
        if unprotected_pos_percent > 0:
            DI = protected_pos_percent / unprotected_pos_percent
        if unprotected_pos_percent == 0.0 and protected_pos_percent == 0.0:
            DI = 1.0
        return DI

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2
