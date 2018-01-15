
class Algorithm():
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

    def numerical_data_only(self):
        """
        Returns True if this algorithm can only handle numerical data as input.
        """
        return False

    def handles_multiple_sensitive_attrs(self):
        """
        Returns True if this algorithm can handle multiple sensitive attributes.
        """
        return False

    def binary_sensitive_attrs_only(self):
        """
        Returns True if this algorithm can only handle sensitive attributes that are binary.
        """
        return True

    def get_name(self):
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}
