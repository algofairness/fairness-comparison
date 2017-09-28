import sys
sys.path.append('/home/h205c/Derek/fairness-comparison')
from algorithms.AbstractAlgorithm import * #AbstractAlgorithm
import BlackBoxAuditing as BBA
from BlackBoxAuditing.model_factories import Weka_SVM, Weka_DecisionTree

class FeldmanAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(FeldmanAlgorithm, self).__init__(*args, **kwargs)

  
  def run(self):
    #d = BBA.load_data(self.data)
    #auditor = BBA.Auditor()
    #auditor.model = Weka_SVM
    #auditor(self.data)

    if self.data == "adult":
      df_res = pd.read_csv('audits/1500997092.53/original_test_data.predictions')
      df_orig = pd.read_csv('audits/1500997092.53/original_test_data.csv')
      val_pos, val_neg = ">50K", "<=50K"
      feldman_protected = []
      for x in df_orig['sex']:
        if x == 'Male':
          feldman_protected.append(1)
        else:
          feldman_protected.append(0)

 
    if self.data == "compas":
      pass

    if self.data == "german":
      df_res = pd.read_csv('audits/1500920731.28/original_test_data.predictions')
      df_orig = pd.read_csv('audits/1500920731.28/original_test_data.csv')
      val_pos, val_neg = "good", "bad"
      feldman_protected = []
      for x in df_orig['personal_status']:
        if 'female' in x:
          feldman_protected.append(0)
        else:
          feldman_protected.append(1)

    if self.data == "retailer":
      df_res = pd.DataFrame()
      df_res['Response'] = [0]
      df_res['Prediction'] = [0]
      val_pos, val_neg = 1,0
      feldman_protected = []
 
    if self.data == "ricci":
      df_res = pd.DataFrame()
      df_res['Response'] = [0]
      df_res['Prediction'] = [0]
      val_pos, val_neg = 1,0
      feldman_protected = []

    feldman_actual = []
    feldman_predicted = []

    for x in df_res['Response']:
      if x == val_pos:
        feldman_actual.append(1)
      elif x == val_neg:
        feldman_actual.append(0)

    for x in df_res['Prediction']:
      if x == val_pos:
        feldman_predicted.append(1)
      elif x == val_neg:
        feldman_predicted.append(0)

    return feldman_actual, feldman_predicted, feldman_protected

def test(data):
  params = {}
  algorithm = FeldmanAlgorithm(data, params)
  print algorithm.run()

if __name__ == "__main__":
  test("german")
  test("adult")
