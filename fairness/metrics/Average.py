from fairness.metrics.Metric import Metric

class Average(Metric):
     """
     Takes the average (mean) of a given list of metrics.  Assumes that if the total over all
     metrics is 0, the returned result should be 1.
     """
     def __init__(self, metrics_list, name):
          Metric.__init__(self)
          self.name = name
          self.metrics = metrics_list

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):

          total = 0.0
          for metric in self.metrics:
               result = metric.calc(actual, predicted, dict_of_sensitive_lists,
                                    single_sensitive_name, unprotected_vals, positive_pred)
               if result != None:
                   total += result

          if total == 0.0:
               return 1.0

          return total / len(self.metrics)

     def is_better_than(self, val1, val2):
          return self.metrics[0].is_better_than(val1, val2)
