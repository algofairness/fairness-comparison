import csv
import pandas as pd
import numpy as np

train_percentage = 1.0/2.0
train_filename = "data/ricci/cleaned-ricci.csv"
test_filename = "data/ricci/ricci.test.csv"
correct_types = [str,float,int,str,float,str]

def clean_ricci_data():
  f = pd.read_csv('data/ricci/RicciDataMod.csv',error_bad_lines=False)
  f.Race.replace(['W','B','H'], [1,0,0], inplace=True)
  f.Position.replace(['Captain','Lieutenant'], [1,0], inplace=True)

  f.to_csv(train_filename)

def load_ricci_data():
  X = []
  y = []
  x_control = []

  data = pd.read_csv(train_filename, error_bad_lines=False)
  data.drop('0',axis=1,inplace=True)
  headers = list(data)
  y = data['Class'].astype(str).tolist()
  x_control = {}
  x_control['Race'] = data['Race'].astype(str).tolist()

  data.drop(['Race','Class'],axis=1, inplace=True)
  X = data.values.astype(str).tolist()

  return np.asarray(X), np.asarray(y), x_control


def test():
  X, y, x_control = load_ricci_data()
  print(X, y, x_control) 

  correct_types = [str, float, int, str, float, str]
  gathered_types = []
  for i, header in enumerate(headers):
    if all( isinstance(row[i],float) for row in train + test ):
      gathered_types.append(float)
    elif all( isinstance(row[i],int) for row in train + test ):
      gathered_types.append(int)
    elif all( isinstance(row[i],str) for row in train + test ):
      gathered_types.append(str)

  print("load_data types are correct? -- ", gathered_types == correct_types)


if __name__=="__main__":
  clean_ricci_data()

