"""Equal opportunity - Protected and unprotected False negative ratio"""
import math
import sys
import numpy

from fairness.metrics.utils import calc_fp_fn
from fairness.metrics.Metric import Metric

class EqOppo_fn_ratio(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_fn_ratio'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
        calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)
        fn_ratio=0.0
        if fn_unprotected > 0:
            fn_ratio= fn_protected/fn_unprotected
        if fn_unprotected == 0.0 and fn_protected == 0.0:
            fn_ratio=1.0
        return fn_ratio
