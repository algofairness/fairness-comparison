from metrics.Metric import Metric
from sklearn.metrics import matthews_corrcoef

class MCC(Metric):
    def __init__(self, actual, predicted):
        Metric.__init__(self, actual, predicted)
        self.name = 'MCC'

    def calc(self):
        return matthews_corrcoef(self.actual, self.predicted)
