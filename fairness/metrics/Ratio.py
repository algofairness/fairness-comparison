from fairness.metrics.Metric import Metric

class Ratio(Metric):
     def __init__(self, metric_numerator, metric_denominator):
          Metric.__init__(self)
          self.numerator = metric_numerator
          self.denominator = metric_denominator
          self.name = self.numerator.get_name() + '_over_' + self.denominator.get_name()

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):
          num = self.numerator.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                                    unprotected_vals, positive_pred)
          den = self.denominator.calc(actual, predicted, dict_of_sensitive_lists,
                                      single_sensitive_name, unprotected_vals, positive_pred)

          if num is None or den is None:
               return None

          if num == 0.0 and den == 0.0:
               return 1.0
          if den == 0.0:
               return 0.0
          return num / den

     def is_better_than(self, val1, val2):
         """
         Assumes that the goal ratio is 1.0.
         """
         dist1 = math.fabs(1.0 - val1)
         dist2 = math.fabs(1.0 - val2)
         return dist1 <= dist2
