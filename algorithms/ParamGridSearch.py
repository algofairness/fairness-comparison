import math

from algorithms.Algorithm import Algorithm

class ParamGridSearch(Algorithm):
    def __init__(self, algorithm, metric):
        Algorithm.__init__(self)
        self.algorithm = algorithm
        self.name = algorithm.get_name() + "-" + metric.get_name()
        # The single metric that will be optimized to for each run of this grid search
        self.metric = metric

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Returns a list of lists of the resulting predictions for all runs of the algorithm in
        the algorithm's search space (accessed via get_param_info).  Given 'params' should be
        empty in the call to this function - the best discovered params are
        returned by mutating the given dictionary.
        """
        all_predictions = []
        search_space = self.algorithm.get_param_info()
        for param_name in search_space:
             ## Note: this only maximizes one parameter at a time - if the maximum involves
             ## two parameters being set, this will not find it.
             for param_val in search_space[param_name]:
                  trial_params = { param_name : param_val }
                  predictions = self.algorithm.run(train_df, test_df, class_attr,
                                                   positive_class_val, sensitive_attrs,
                                                   single_sensitive,
                                                   privileged_vals, trial_params)
                  all_predictions.append( (param_name, param_val, predictions) )
        best_predictions = self.find_best(all_predictions, train_df, test_df, class_attr,
                                          positive_class_val, sensitive_attrs, single_sensitive,
                                          privileged_vals, params)
        return best_predictions

    def find_best(self, all_predictions, train_df, test_df, class_attr, positive_class_val,
                  sensitive_attrs, single_sensitive, privileged_vals, params):
        actual = test_df[class_attr]
        sensitive = test_df[single_sensitive].values.tolist()

        best_val = None
        best = None
        best_name = None
        for param_name, param_val, predictions in all_predictions:
             val = self.metric.calc(actual, predictions, sensitive, privileged_vals,
                                    positive_class_val)
             if best_val == None or self.metric.is_better_than(val, best_val):
                  best = predictions
                  best_name = param_name
                  best_val = param_val
        self.reset_params(best_name, best_val, params)
        return best

    def reset_params(self, param_name, param_val, param_dict):
        for old_name in param_dict:
            del param_dict[old_name]
        param_dict[param_name] = param_val

    def numerical_data_only(self):
        return self.algorithm.numerical_data_only()

    def handles_multiple_sensitive_attrs(self):
        """
        Returns True if this algorithm can handle multiple sensitive attributes.
        """
        return self.algorithm.handles_multiple_sensitive_attrs() 

