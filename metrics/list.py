from metrics.Accuracy import Accuracy
from metrics.BCROutcome import BCROutcome
from metrics.CV import CV
from metrics.DisparateImpact import DisparateImpact
from metrics.MCC import MCC
from metrics.TNR import TNR
from metrics.TPR import TPR

METRICS = [ Accuracy(), TPR(), TNR(), BCROutcome(), MCC(), DisparateImpact(), CV() ]
