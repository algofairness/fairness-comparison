import os
import sys
from algorithms.AbstractAlgorithm import * #AbstractAlgorithm
import BlackBoxAuditing as BBA
from BlackBoxAuditing.model_factories import Weka_SVM, Weka_DecisionTree
from datetime import datetime

class FeldmanAlgorithm(AbstractAlgorithm):
  def __init__(self, *args, **kwargs):
    super(FeldmanAlgorithm, self).__init__(*args, **kwargs)

  
  def run(self):
    startTime = datetime.now()
    if self.data == "ricci":
      datafile = 'data/ricci/cleaned-ricci.csv'
      export_to = 'audits/ricci'
      correct_types = [str,str,str,str,str,str]
      train_percentage = 1.0/2.0
      response_header = "Class"
      features_to_ignore = ["Position"]
    
    if self.data == "german":
      datafile = 'data/german/german_numeric_sex_encoded_fixed.csv'
      export_to = 'audits/german'
      correct_types = [str,str,str,str,str,str,str]
      train_percentage = 2.0/3.0
      response_header = "Credit"
      features_to_ignore = []

    if self.data == "adult":
      datafile = 'data/adult/adult-all-numerical-converted.csv'
      export_to = 'audits/adult'
      correct_types = [str,str,str,str,str,str,str]
      train_percentage = 2.0/3.0
      response_header = "income_per_year"
      features_to_ignore = []

    if self.data == "retailer":
      datafile = 'data/retailer/cleaned-retailer.csv'
      #datafile = 'data/retailer/small-cleaned-retailer.csv'
      export_to = 'audits/retailer'
      correct_types = [str for i in range(27)]
      train_percentage = 2.0/3.0
      response_header = "hired" 
      features_to_ignore = []
   
    data = BBA.load_from_file(datafile, testdata=None, correct_types=correct_types, train_percentage=train_percentage, response_header=response_header, features_to_ignore=features_to_ignore, missing_data_symbol="")
    #data = BBA.load_data(self.data)
    auditor = BBA.Auditor()
    auditor.model = self.params['model'] 
    auditor(data, output_dir=export_to,features_to_audit=[self.sensitive_attr])

    if self.data == "adult":
      df = pd.read_csv('audits/adult/sex.audit.repaired_0.9999999999999999.predictions')
 
    if self.data == "compas":
      pass

    if self.data == "german":
      df = pd.read_csv('audits/german/sex.audit.repaired_0.9999999999999999.predictions')

    if self.data == "retailer":
      df = pd.read_csv('audits/retailer/urace_orig.audit.repaired_0.9999999999999999.predictions')
 
    if self.data == "ricci":
      df = pd.read_csv('audits/ricci/Race.audit.repaired_0.9999999999999999.predictions')

    '''
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
    '''
    feldman_actual = df['Response']
    feldman_predicted = df['Prediction']
    feldman_protected = df['Pre-Repaired Feature']
    feldman_time = datetime.now() - startTime
    
    return feldman_actual, feldman_predicted, feldman_protected, feldman_time

def test(data):
  params = {}
  algorithm = FeldmanAlgorithm(data, params)
  print(algorithm.run())

if __name__ == "__main__":
  test("german")
  test("adult")
