from metrics.Accuracy import Accuracy
from metrics.MCC import MCC
from metrics.DisparateImpact import DisparateImpact
from metrics.TPR import TPR
from metrics.TNR import TNR

METRICS = [ Accuracy(), TPR(), TNR(), MCC(), DisparateImpact() ]
