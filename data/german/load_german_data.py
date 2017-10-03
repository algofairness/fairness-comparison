import numpy as np
from sklearn import preprocessing
import os

def load_german_data(filename):
  X = []
  y = []
  x_control = []
  headers = []
  f = open(filename).readlines()
  headers = f.pop(0)[:-1].split(',')
  sensitive_index = headers.index("gender") 
  for line in f:
    line = line[:-1].split(',')

    # Get class label
    class_label = line[-1]
    y.append(class_label) 

    # Get sensitive variable
    x_control.append(line[sensitive_index])

    # Add everything else to X
    rest_of_line = line[:8]+line[9:-1]
    X.append(rest_of_line)

  X = np.array(X)
  y = np.array(y)
  x_control = {"sex": x_control}
  print( X, y, x_control)
    

def test():
  load_german_data("german_numeric_sex_encoded_fixed.csv")

if __name__ == "__main__":
  test()
