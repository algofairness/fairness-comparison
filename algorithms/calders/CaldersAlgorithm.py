import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from algorithms.AbstractAlgorithm import *
from algorithms.calders.two_naive_bayes import *

class CaldersAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(CaldersAlgorithm, self).__init__(*args, **kwargs)

  def run(self):
    c2nb_protected_predicted, c2nb_protected_actual, c2nb_favored_predicted, c2nb_favored_actual = run_two_naive_bayes(0.0, self.filename, self.x_train, self.y_train, self.x_control_train, self.x_test, self.y_test, self.x_control_test, self.sensitive_attr)
    c2nb_protected_protected = [0] * len(c2nb_protected_predicted)
    c2nb_favored_protected   = [1] * len(c2nb_favored_predicted)

    # Combine into one data set with protected and unprotected
    c2nb_predicted = c2nb_protected_predicted + c2nb_favored_predicted
    c2nb_actual = c2nb_protected_actual + c2nb_favored_actual
    c2nb_protected = c2nb_protected_protected + c2nb_favored_protected

    return c2nb_actual, c2nb_predicted, c2nb_protected

def test(data):
  params = {}
  algorithm = CaldersAlgorithm(data, params)
  print algorithm.run()

if __name__ == "__main__":
  test('german')
  test('adult')
