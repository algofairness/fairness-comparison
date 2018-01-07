from metrics.Accuracy import Accuracy
from metrics.MCC import MCC
from metrics.DisparateImpact import DisparateImpact

METRICS = [ Accuracy(), MCC() ]
FAIRNESS_METRICS = [ DisparateImpact() ]
