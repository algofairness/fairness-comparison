from metrics.Accuracy import Accuracy
from metrics.Average import Average
from metrics.FilterSensitive import FilterSensitive
from metrics.Metric import Metric

class SensitiveAccuracy(Metric):
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
          self.name = 'Accuracy'  # to be modified as this metric is expanded

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):
          sfilter = FilterSensitive(Accuracy())
          sfilter.set_sensitive_to_filter(self.sensitive_attr, self.sensitive_val)
          return sfilter.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                              unprotected_vals, positive_pred)

     def expand_per_dataset(self, dataset, sensitive_dict):
          objects_list = []
          for sensitive in dataset.get_sensitive_attributes_with_joint():
               objects_list += make_metric_objects(sensitive, sensitive_dict)
          return objects_list

     def set_sensitive_to_filter(self, sensitive_name, sensitive_val):
          """
          Set the attribute and value to filter, i.e., to calculate this metric for.
          """
          self.sensitive_attr = sensitive_name
          self.sensitive_val = sensitive_val
          self.name = str(sensitive_val) + "-" + self.name

def make_metric_objects(sensitive_name, sensitive_values):
     objs_list = []
     for val in sensitive_values[sensitive_name]:
         obj = SensitiveAccuracy()
         obj.set_sensitive_to_filter(sensitive_name, val)
         objs_list.append(obj)
     avg = Average(objs_list, sensitive_name + '-' + SensitiveAccuracy().get_name())
     return objs_list + [avg]
