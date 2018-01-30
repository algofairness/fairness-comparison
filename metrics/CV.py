import sys
import numpy
import math

from metrics.utils import calc_pos_protected_percents
from metrics.Metric import Metric

class CV(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'CV'

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
        unprotected_pos_percent, protected_pos_percent = \
            calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred)
        CV = math.fabs(unprotected_pos_percent - protected_pos_percent)
        return 1.0 - CV
