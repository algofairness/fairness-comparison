from fairness.metrics.Metric import Metric

class FilterSensitive(Metric):
     def __init__(self, metric):
          Metric.__init__(self)
          self.metric = metric
          self.name = metric.get_name()

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):

          sensitive = dict_of_sensitive_lists[self.sensitive_for_metric]
          actual_sens = \
              [act for act, sens in zip(actual, sensitive) if sens == self.sensitive_filter]
          predicted_sens = \
              [pred for pred, sens in zip(predicted, sensitive) if sens == self.sensitive_filter]
          sensitive_sens = \
             [sens for sens in sensitive if sens == self.sensitive_filter]

          filtered_dict = {}
          for sens_val in dict_of_sensitive_lists:
              other_sensitive = dict_of_sensitive_lists[sens_val]
              filtered =  \
                  [s for s, sens in zip(other_sensitive, sensitive) if sens == self.sensitive_filter]
              filtered_dict[sens_val] = filtered

          if len(actual_sens) < 1:
              return None

          return self.metric.calc(actual_sens, predicted_sens, filtered_dict, single_sensitive_name,
                                  unprotected_vals, positive_pred)

     def set_sensitive_to_filter(self, sensitive_name, sensitive_val):
          """
          Sets the specific sensitive value to filter based on.  The given metric will be
          calculated only with respect to the actual and predicted values that have this sensitive
          value as part of that item.

          sensitive_name        sensitive attribute name (e.g., 'race')
          sensitive_val         specific sensitive value (e.g., 'white')
          """
          self.name += str(sensitive_val)
          self.sensitive_filter = sensitive_val
          self.sensitive_for_metric = sensitive_name

