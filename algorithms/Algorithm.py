
class Algorithm():
    def __init__(self):
        pass

    def run(self, train_df, test_df, sensitive_attrs, params):
        """
        Runs the algorithm and returns the actual classifications, predicted classifications,
        and a vector of associated sensitive attributes.
        """
        raise NotImplementedError("run() in Algorithm is not implemented")
