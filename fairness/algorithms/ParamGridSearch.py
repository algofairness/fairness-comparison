import math

from fairness.algorithms.Algorithm import Algorithm

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
                  try:
                      predictions, trash = \
                          self.algorithm.run(train_df, test_df, class_attr, positive_class_val,
                                             sensitive_attrs, single_sensitive, privileged_vals,
                                             trial_params)
                      all_predictions.append( (param_name, param_val, predictions) )
                  except Exception as e:
                      print("run for parameters %s failed: %s" % (params, e))
        best_predictions = self.find_best(all_predictions, train_df, test_df, class_attr,
                                          positive_class_val, sensitive_attrs, single_sensitive,
                                          privileged_vals, params)
        return best_predictions, all_predictions

    def find_best(self, all_predictions, train_df, test_df, class_attr, positive_class_val,
                  sensitive_attrs, single_sensitive, privileged_vals, params):
        if len(all_predictions) == 0:
            raise Exception(
                "No run in the parameter grid search succeeded - failing run of algorithm")
        actual = test_df[class_attr]
        dict_sensitive = {}
        for sens in sensitive_attrs:
             dict_sensitive[sens] = test_df[sens].values.tolist()

        best_val = None
        best = None
        best_name = None
        best_metric = None
        for param_name, param_val, predictions in all_predictions:
             val = self.metric.calc(actual, predictions, dict_sensitive, single_sensitive,
                                    privileged_vals, positive_class_val)
             if best_val == None or self.metric.is_better_than(val, best_metric):
                  best = predictions
                  best_name = param_name
                  best_val = param_val
                  best_metric = val
        self.reset_params(best_name, best_val, params)
        return best

    def reset_params(self, param_name, param_val, param_dict):
        for old_name in param_dict:
            del param_dict[old_name]
        param_dict[param_name] = param_val

    def get_supported_data_types(self):
        return self.algorithm.get_supported_data_types()

    def handles_multiple_sensitive_attrs(self):
        """
        Returns True if this algorithm can handle multiple sensitive attributes.
        """
        return self.algorithm.handles_multiple_sensitive_attrs()

