
class Algorithm():
    """
    This is the base class for all implemented algorithms.  New algorithms should extend this
    class, implement run below, and set self.name in the init method.  Other optional methods to
    implement are described below.
    """

    def __init__(self):
        pass

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        params: a dictionary mapping from algorithm-specific parameter names to the desired values.
        If the implementation of run uses different values, these should be modified in the params
        dictionary as a way of returning the used values to the caller.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df.  If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        TODO: figure out how to indicate that an algorithm that can handle multiple sensitive
        attributes should do so now.
        """
        raise NotImplementedError("run() in Algorithm is not implemented")

    def get_param_info(self):
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to
        be explored.  This function should only be implemented if the algorithm has specific
        parameters that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):
        """
        Returns a set of datatypes which this algorithm can process.
        """
        raise NotImplementedError("get_supported_data_types() in Algorithm is not implemented")

    def get_name(self):
        """
        Returns the name for the algorithm.  This must be a unique name, so it is suggested that
        this name is simply <firstauthor>.  If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>.  This name will appear in the resulting
        CSVs and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}
