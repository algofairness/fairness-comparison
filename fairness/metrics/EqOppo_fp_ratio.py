""" Equal opportunity - Protected and unprotected False postives ratio"""
import math
import sys
import numpy

from fairness.metrics.utils import calc_fp_fn
from fairness.metrics.Metric import Metric

class EqOppo_fp_ratio(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_fp_ratio'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]

        fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
        calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)
        fp_ratio=0.0
        if fp_unprotected > 0:
            fp_ratio= fp_protected/fp_unprotected
        if fp_unprotected == 0.0 and fp_protected == 0.0:
            fp_ratio=1.0
        return fp_ratio
