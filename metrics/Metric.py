class Metric:
    def __init__(self):
        self.name = 'Name not implemented'  ## This should be replaced in implemented metrics.

    def calc(self, actual, predicted):
        raise NotImplementedError("calc() in Metric is not implemented")

    def get_name(self):
        """
        Returns a name for the metric.  This will be used as the key for a dictionary and will
        also be printed to the final output file.
        """
        return self.name
