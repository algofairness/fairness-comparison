""" Equal opportunity - Protected and unprotected False negative difference"""
import math
import sys
import numpy

from metrics.utils import calc_fp_fn
from metrics.Metric import Metric

class EqOppo_fn_diff(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_fn_diff'

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
    	fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
    	calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)

    	fn_diff= math.fabs(fn_protected-fn_unprotected)

    	return fn_diff
