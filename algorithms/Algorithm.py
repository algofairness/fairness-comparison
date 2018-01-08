
class Algorithm():
    def __init__(self):
        pass

    def run(self, train_df, test_df, class_attr, sensitive_attrs, single_sensitive, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given 
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        TODO: figure out how to indicate that an algorithm that can handle multiple sensitive
        attributes should do so now.
        """
        raise NotImplementedError("run() in Algorithm is not implemented")

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

    def get_name(self):
        return self.name
