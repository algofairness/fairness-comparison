from metrics.Accuracy import Accuracy
from metrics.CV import CV
from metrics.DisparateImpact import DisparateImpact
from metrics.MCC import MCC
from metrics.TNR import TNR
from metrics.TPR import TPR

METRICS = [ Accuracy(), TPR(), TNR(), MCC(), DisparateImpact(), CV(), EqOppo_fp_diff(), EqOppo_fp_ratio(), EqOppo_fn_diff(), EqOppo_fn_ratio()]
