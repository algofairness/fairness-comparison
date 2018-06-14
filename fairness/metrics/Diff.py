from fairness.metrics.Metric import Metric

class Diff(Metric):
     def __init__(self, metric1, metric2):
          Metric.__init__(self)
          self.metric1 = metric1
          self.metric2 = metric2
          self.name = "diff:" + self.metric1.get_name() + 'to' + self.metric2.get_name()

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):
          m1 = self.metric1.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                                 unprotected_vals, positive_pred)
          m2 = self.metric2.calc(actual, predicted, dict_of_sensitive_lists,
                                 single_sensitive_name, unprotected_vals, positive_pred)

          if m1 is None or m2 is None:
               return None

          diff = m1 - m2
          return 1.0 - diff

     def is_better_than(self, val1, val2):
         """
         Assumes that 1.0 is the goal value.
         """
         dist1 = math.fabs(1.0 - val1)
         dist2 = math.fabs(1.0 - val2)
         return dist1 <= dist2
