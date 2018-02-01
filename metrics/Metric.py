class Metric:
    def __init__(self):
        self.name = 'Name not implemented'  ## This should be replaced in implemented metrics.
        self.iter_counter = 0

    def __iter__(self):
        self.iter_counter = 0
        return self

    def __next__(self):
        self.iter_counter += 1
        if self.iter_counter > 1:
            raise StopIteration
        return self

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
        """
        Inputs: the actual results on the test set, the predicted results, a vector of the associated
        sensitive values, a list of the unprotected values for the sensitive categories, and 
        the positive value of the prediction task.  The actual and predicted results and the 
        sensitive attributes vector should have the same length (the length of the test set).
        """
        raise NotImplementedError("calc() in Metric is not implemented")

    def get_name(self):
        """
        Returns a name for the metric.  This will be used as the key for a dictionary and will
        also be printed to the final output file.
        """
        return self.name

    def is_better_than(self, val1, val2):
        """
        Compares the two given values that were calculated by this metric and returns true if 
        val1 is better than val2, false otherwise.
        """
        return val1 > val2

    def expand_per_dataset(self, dataset):
        """
        Optionally allows the expansion of the metric into a returned list of metrics based on the
        dataset, e.g., where there is one metric per sensitive attribute given.
        """
        return self
