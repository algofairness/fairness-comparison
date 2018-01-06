
class Algorithm():
    def __init__(self):
        pass

    def run(self, train_df, test_df, class_attr, sensitive_attrs, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.
        """
        raise NotImplementedError("run() in Algorithm is not implemented")

    def numerical_data_only(self):
        """
        Returns True if this algorithm can only handle numerical data as input.
        """
        return False 
