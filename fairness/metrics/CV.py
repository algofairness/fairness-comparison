import sys
import numpy
import math

from fairness.metrics.utils import calc_pos_protected_percents
from fairness.metrics.Metric import Metric

class CV(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'CV'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_pos_percent, protected_pos_percent = \
            calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred)
        CV = unprotected_pos_percent - protected_pos_percent
        return 1.0 - CV

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2
