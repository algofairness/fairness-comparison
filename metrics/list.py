import numpy

from metrics.Accuracy import Accuracy
from metrics.BCROutcome import BCROutcome
from metrics.BCRSensitive import BCRSensitive
from metrics.CV import CV
from metrics.DIAvgAll import DIAvgAll
from metrics.DIBinary import DIBinary
from metrics.EqOppo_fn_diff import EqOppo_fn_diff
from metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from metrics.EqOppo_fp_diff import EqOppo_fp_diff
from metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from metrics.MCC import MCC
from metrics.SensitiveAccuracy import SensitiveAccuracy
from metrics.TNR import TNR
from metrics.TPR import TPR

METRICS = [ Accuracy(), TPR(), TNR(), BCROutcome(), MCC(),      # accuracy metrics
            DIBinary(), DIAvgAll(), CV(), BCRSensitive(),       # fairness metrics
            SensitiveAccuracy() ]
#            EqOppo_fn_diff(), EqOppo_fp_diff(), EqOppo_fn_ratio(), EqOppo_fp_ratio() ]

def get_metrics(dataset, sensitive_dict):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict)
    return metrics
