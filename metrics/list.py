from metrics.Accuracy import Accuracy
from metrics.MCC import MCC
from metrics.DisparateImpact import DisparateImpact
from metrics.CV import CV

METRICS = [ Accuracy(), MCC(), DisparateImpact(), CV() ]
