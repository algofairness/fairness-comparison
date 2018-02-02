import numpy

from metrics.Accuracy import Accuracy
from metrics.BCROutcome import BCROutcome
from metrics.BCRSensitive import BCRSensitive
from metrics.CV import CV
from metrics.DIAvgAll import DIAvgAll
from metrics.DIBinary import DIBinary
from metrics.MCC import MCC
from metrics.TNR import TNR
from metrics.TPR import TPR

METRICS = [ Accuracy(), TPR(), TNR(), BCROutcome(), MCC(),      # accuracy metrics
            DIBinary(), DIAvgAll(), CV(), BCRSensitive() ]        # fairness metrics

def get_metrics(dataset):
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset)
    return metrics
