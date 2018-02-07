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

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        """
        actual                          a list of the actual results on the test set
        predicted                       a list of the predicted results
        dict_of_sensitive_lsits         dict mapping sensitive attr names to list of sensitive vals
        single_sensitive_name           sensitive name (dict key) for the sensitive attr being
                                        focused on by this run of the algorithm
        unprotected_vals                a list of the unprotected values for all sensitive attrs
        positive_pred                   the positive value of the prediction task.

        returns                         the calculated result for this metric

        The actual and predicted results and the sensitive attribute lists in the dict should have
        the same length (the length of the test set).

        If there is an error and the metric can not be calculated (e.g., no data is passed in), the
        metric returns None.
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

    def expand_per_dataset(self, dataset, sensitive_dict, tag):
        """
        Optionally allows the expansion of the metric into a returned list of metrics based on the
        dataset, e.g., where there is one metric per sensitive attribute given, and a dictionary
        mapping sensitive attributes to all seen sensitive values from the data.
        """
        return self
