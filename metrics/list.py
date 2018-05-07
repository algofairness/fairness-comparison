import numpy

from metrics.Accuracy import Accuracy
from metrics.BCR import BCR
from metrics.CalibrationNeg import CalibrationNeg
from metrics.CalibrationPos import CalibrationPos
from metrics.CV import CV
from metrics.DIAvgAll import DIAvgAll
from metrics.DIBinary import DIBinary
from metrics.EqOppo_fn_diff import EqOppo_fn_diff
from metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from metrics.EqOppo_fp_diff import EqOppo_fp_diff
from metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from metrics.FNR import FNR
from metrics.FPR import FPR
from metrics.MCC import MCC
from metrics.SensitiveMetric import SensitiveMetric
from metrics.TNR import TNR
from metrics.TPR import TPR


METRICS = [ Accuracy(), TPR(), TNR(), BCR(), MCC(),        # accuracy metrics
            DIBinary(), DIAvgAll(), CV(),                  # fairness metrics
            SensitiveMetric(Accuracy), SensitiveMetric(TPR), SensitiveMetric(TNR),
            SensitiveMetric(FPR), SensitiveMetric(FNR),
            SensitiveMetric(CalibrationPos), SensitiveMetric(CalibrationNeg) ]

def get_metrics(dataset, sensitive_dict, tag):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)
    return metrics
