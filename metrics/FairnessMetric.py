from Metric import Metric

class FairnessMetric(Metric):
    def __init__(self, actual, predicted, protected):
        Metric.__init__(actual, predicted)
        self.protected = protected

    def calc():
        pass 
