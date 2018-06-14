from fairness.metrics.Metric import Metric

class CalibrationNeg(Metric):
     def __init__(self):
          Metric.__init__(self)
          self.name = 'calibration-'

     def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
              unprotected_vals, positive_pred):
          total_pred_negative = 0.0
          act_correct = 0.0
          for act, pred in zip(actual, predicted):
               if pred != positive_pred:
                   total_pred_negative += 1
                   if act == positive_pred:
                        act_correct += 1
          if act_correct == 0.0 and total_pred_negative == 0.0:
               return 1.0
          if total_pred_negative == 0.0:
               return 0.0
          return act_correct / total_pred_negative
