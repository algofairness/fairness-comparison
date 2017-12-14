class Metric:
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predited = predicted

    def calc(self):
        raise NotImplementedError("calc() in Metric is not implemented")
