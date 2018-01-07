from metrics.Metric import Metric
from sklearn.metrics import matthews_corrcoef

class MCC(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'MCC'

    def calc(self, actual, predicted):
        return matthews_corrcoef(actual, predicted)
