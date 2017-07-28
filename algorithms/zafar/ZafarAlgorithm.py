import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from algorithms.AbstractAlgorithm import *
from misc.zafar_classifier import *
import algorithms.zafar.fair_classification.utils as ut
import algorithms.zafar.fair_classification.loss_funcs as lf

class ZafarAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(ZafarAlgorithm, self).__init__(*args, **kwargs)

  def run(self):
    print("Running Zafar...")
    sensitive_attrs = [str(self.sensitive_attr)]
    loss_function = lf._logistic_loss

    # Defaults to None
    if "gamma" in self.params.keys():
      gamma = self.params["gamma"]
    else:
      gamma = None

    # Defaults to 0
    if "apply_accuracy_constraint" in self.params.keys():
      apply_accuracy_constraint = self.params["apply_accuracy_constraint"]
    else:
      apply_accuracy_constraint = 0  

    # Defaults to 0
    if "apply_fairness_constraints" in self.params.keys():
      apply_fairness_constraints = self.params["apply_fairness_constraints"]
    else:
      apply_fairness_constraints = 0

    # Defaults to 0
    if "sep_constraint" in self.params.keys():
      sep_constraint = self.params["sep_constraint"]
    else:
      sep_constraint = 0 

    # Defaults to {}
    if "sensitive_attrs_to_cov_thresh" in self.params.keys():
      sensitive_attrs_to_cov_thresh = self.params["sensitive_attrs_to_cov_thresh"]
    else:
      sensitive_attrs_to_cov_thresh = {}

    w = ut.train_model(self.x_train, self.y_train, self.x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
    distances_boundary_test = (np.dot(self.x_test, w)).tolist()
    predictions = np.sign(distances_boundary_test)

    fixed_y_test = []
    fixed_predictions = []

    for x in self.y_test:
      if x == -1:
        fixed_y_test.append(0)
      elif x == 1:
        fixed_y_test.append(1)
      elif x == 0:
        fixed_y_test.append(0)
      else:
        print "Incorrect value in class values"

    for x in predictions:
      if x == -1:
        fixed_predictions.append(0)
      elif x == 1:
        fixed_predictions.append(1)
      elif x == 0:
        fixed_predictions.append(0)
      else:
        print "Incorrect value in class values"

    zafar_actual, zafar_predicted, zafar_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]
    return zafar_actual, zafar_predicted, zafar_protected

def test(data):
  params = {}
  algorithm = ZafarAlgorithm(data, params)
  print "Unconstrained: ", algorithm.run()

  params["apply_fairness_constraints"] = 1
  params["sensitive_attrs_to_cov_thresh"] = {algorithm.sensitive_attr:0}
  algorithm = ZafarAlgorithm(data, params)
  print "Opt for accuracy: ", algorithm.run()

  params["apply_accuracy_constraint"] = 1
  params["apply_fairness_constraints"] = 0
  params["sensitive_attrs_to_cov_thresh"] = {}
  params["gamma"] = 0.5
  algorithm = ZafarAlgorithm(data, params)
  print "Opt for fairness: ", algorithm.run()

  params["sep_constraint"] = 1
  params["gamma"] = 1000.0
  algorithm = ZafarAlgorithm(data, params)
  print "No pos classification errors: ", algorithm.run()

if __name__ == "__main__":
  test("german")
  test("adult")
