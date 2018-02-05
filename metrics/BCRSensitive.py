from metrics.Accuracy import Accuracy
from metrics.Metric import Metric

class BCRSensitive(Metric):
     """
     This measure takes the average accuracy per sensitive value.  It is unweighted in the sense
     that each sensitive value's accuracy is treated equally in the average.  This measure is
     designed to catch the scenario when misclassifying all Native-Americans but having high
     accuracy (say, 100%) on everyone else causes an algorithm to have 98% accuracy because
     Native-Americans make up about 2% of the U.S. population.  In this scenario, assuming the
     listed sensitive values were Native-American and not-Native-American, this metric would
     return 0.5.  Given more than two sensitive values, it will return the average over all of the
     per-value accuracies.
     """
     def __init__(self):
          Metric.__init__(self)
          self.name = 'BCR'  # This will be modified per sensitive attribute considered.

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):
          total = 0.0
          sensitive = dict_of_sensitive_lists[self.sensitive_for_metric]
          sensitive_values = list(set(sensitive))
          for sens_val in sensitive_values:
              actual_sens = \
                  [act for act, sens in zip(actual, sensitive) if sens_val == sens]
              predicted_sens = \
                  [pred for pred, sens in zip(predicted, sensitive) if sens_val == sens]
              sensitive_sens = \
                  [sens for sens in sensitive if sens_val == sens]
              acc = Accuracy()
              acc_sens = acc.calc(actual_sens, predicted_sens, dict_of_sensitive_lists,
                                  single_sensitive_name, unprotected_vals, positive_pred)
              total += acc_sens
          return total / len(sensitive_values)

     def expand_per_dataset(self, dataset, sensitive_dict):
          objects_list = []
          for sensitive in dataset.get_sensitive_attributes_with_joint():
               objects_list += make_metric_object(sensitive)
          return objects_list

     def set_sensitive_for_metric(self, sensitive_name):
          """
          Sets the specific sensitive attr that this metric will be calculated with respect to,
          i.e., the sensitive values that it will take the average over.  Note that this may be
          different than the sensitive value that the algorithm being analyzed is attempting to be
          fair in terms of.
          """
          self.name += sensitive_name
          self.sensitive_for_metric = sensitive_name

def make_metric_object(sensitive_name):
     obj = BCRSensitive()
     obj.set_sensitive_for_metric(sensitive_name)
     return obj

