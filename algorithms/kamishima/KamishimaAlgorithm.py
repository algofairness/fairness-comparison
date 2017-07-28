import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from algorithms.AbstractAlgorithm import *
from misc.prejudice_regularizer import *

class KamishimaAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(KamishimaAlgorithm, self).__init__(*args, **kwargs)

  def run(self):
    # Defaults to 1
    if "eta" in self.params.keys():
      eta = self.params["eta"]
    else:
      eta = 1
    
    x_train_with_sensitive_feature = []

    for i in range(0, len(self.x_train)):
      val = self.x_control_train[self.sensitive_attr][i]
      feature_array = np.append(self.x_train[i], val)
      x_train_with_sensitive_feature.append(feature_array)

    x_train_with_sensitive_feature = np.array(x_train_with_sensitive_feature)

    x_test_with_sensitive_feature = []
    for i in range(0, len(self.x_test)):
      val = self.x_control_test[self.sensitive_attr][i]
      feature_array = np.append(self.x_test[i], val)
      x_test_with_sensitive_feature.append(feature_array)

    x_test_with_sensitive_feature = np.array(x_test_with_sensitive_feature)

    y_classified_results = train_classify(self.sensitive_attr, self.name, x_train_with_sensitive_feature, self.y_train, x_test_with_sensitive_feature, self.y_test, 1, eta, self.x_control_test)
    fixed_y_test = []
    for j in self.y_test:
        if j == 1.0:
            fixed_y_test.append(1)
        elif j == -1.0 or j == 0.0:
            fixed_y_test.append(0)
        else:
            print "Invalid class value in y_control_test"

    kam_actual, kam_predicted, kam_protected = fixed_y_test, y_classified_results, self.x_control_test[self.sensitive_attr]

    return kam_actual, kam_predicted, kam_protected

def test(data):
  params = {}
  params["eta"] = 1
  algorithm = KamishimaAlgorithm(data, params) 
  print algorithm.run()

  params["eta"] = 30
  algorithm = KamishimaAlgorithm(data, params)
  print algorithm.run()

if __name__ == "__main__":
  test("german")
  test("adult")
