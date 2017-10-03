import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from algorithms.AbstractAlgorithm import *
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#class SKLearnModelFactory(AbstractAlgorithm):
class GenAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(GenAlgorithm, self).__init__(*args, **kwargs)
    self.models = {SVC():'svm', GaussianNB():'nb', LogisticRegression():'lr'}

  def run(self):
    for modeltype in self.models.keys():
      model = modeltype
      model.fit(self.x_train, self.y_train)
      predictions = model.predict(self.x_test)
      fixed_predictions = []
      fixed_y_test = []

      for j in range(0, len(predictions)):
        if predictions[j] == 0.0:
          fixed_predictions.append(0)
        elif predictions[j] == 1.0:
          fixed_predictions.append(1)

      for j in range(0, len(self.y_test)):
        if self.y_test[j] == 0.0:
          fixed_y_test.append(0)
        elif self.y_test[j] == 1.0:
          fixed_y_test.append(1)
      
      if self.models[modeltype] == 'svm':
        svm_actual, svm_predicted, svm_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]
      if self.models[modeltype] == 'nb':
        nb_actual, nb_predicted, nb_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]
      if self.models[modeltype] == 'lr':
        lr_actual, lr_predicted, lr_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]

    return svm_actual, svm_predicted, svm_protected, nb_actual, nb_predicted, nb_protected, lr_actual, lr_predicted, lr_protected

      #actual, predicted, protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]

"""
  def run(self):
    #SVM
    clf = SVC()
    clf.fit(self.x_train, self.y_train)
    predictions = clf.predict(self.x_test)
    fixed_predictions = []
    fixed_y_test = []

    for j in range(0, len(predictions)):
      if predictions[j] == 0.0:
        fixed_predictions.append(0)
      elif predictions[j] == 1.0:
        fixed_predictions.append(1)

    for j in range(0, len(self.y_test)):
      if self.y_test[j] == 0.0:
        fixed_y_test.append(0)
      elif self.y_test[j] == 1.0:
        fixed_y_test.append(1)

    svm_actual, svm_predicted, svm_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]

    #NB
    nb = GaussianNB()
    nb.fit(self.x_train, self.y_train)
    predictions = nb.predict(self.x_test)
    fixed_predictions = []
    fixed_y_test = []

    for j in range(0, len(predictions)):
      if predictions[j] == 0.0:
        fixed_predictions.append(0)
      elif predictions[j] == 1.0:
        fixed_predictions.append(1)

    for j in range(0, len(self.y_test)):
      if self.y_test[j] == 0.0:
        fixed_y_test.append(0)
      elif self.y_test[j] == 1.0:
        fixed_y_test.append(1)

    nb_actual, nb_predicted, nb_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]

    #LR
    lr = LogisticRegression()
    lr.fit(self.x_train, self.y_train)
    predictions = lr.predict(self.x_test)
    fixed_predictions = []
    fixed_y_test = []

    for j in range(0, len(predictions)):
      if predictions[j] == 0.0:
        fixed_predictions.append(0)
      elif predictions[j] == 1.0:
        fixed_predictions.append(1)

    for j in range(0, len(self.y_test)):
      if self.y_test[j] == 0.0:
        fixed_y_test.append(0)
      elif self.y_test[j] == 1.0:
        fixed_y_test.append(1)

    lr_actual, lr_predicted, lr_protected = fixed_y_test, fixed_predictions, self.x_control_test[self.sensitive_attr]

    return svm_actual, svm_predicted, svm_protected, nb_actual, nb_predicted, nb_protected, lr_actual, lr_predicted, lr_protected
"""
def test(data):
  params = {}
  algorithm = GenAlgorithm(data, params)
  print(algorithm.run())

if __name__ == "__main__":
  test('german')
  test('adult')
