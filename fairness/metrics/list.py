import numpy

from fairness.metrics.Accuracy import Accuracy
from fairness.metrics.BCR import BCR
from fairness.metrics.CalibrationNeg import CalibrationNeg
from fairness.metrics.CalibrationPos import CalibrationPos
from fairness.metrics.CV import CV
from fairness.metrics.DIAvgAll import DIAvgAll
from fairness.metrics.DIBinary import DIBinary
from fairness.metrics.EqOppo_fn_diff import EqOppo_fn_diff
from fairness.metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from fairness.metrics.EqOppo_fp_diff import EqOppo_fp_diff
from fairness.metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from fairness.metrics.FNR import FNR
from fairness.metrics.FPR import FPR
from fairness.metrics.MCC import MCC
from fairness.metrics.SensitiveMetric import SensitiveMetric
from fairness.metrics.TNR import TNR
from fairness.metrics.TPR import TPR

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

def add_metric(metric):
    METRICS.append(metric)
