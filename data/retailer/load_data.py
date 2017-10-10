import datetime
from datetime import date
import time
import pandas as pd
import numpy as np
import csv
import random

train_percentage = 2.0/3.0
train_filename = "data/retailer/cleaned-retailer.csv"
#train_filename = "data/retailer/small-cleaned-retailer.csv"
test_filename = "data/retailer/retailer.test.csv"
max_entries = None
reqs = [int]*25
correct_types = [int,int,int,str,str,float,int,float,str] + reqs

def clean_retailer_data():
  f = pd.read_csv('data/retailer/retailer.csv',error_bad_lines=False)
  keep_cols = ['usite','apscustpersonid','azip','urace_orig','udateofbirth','ugender','szip','csvr2','hired']
  for col in list(f):
    if 'req_' in col:
      keep_cols.append(col)
  data = f[keep_cols].dropna()
  data.urace_orig.replace(['White','Black or African American', 'Black/African-American', 'Asian', 'Amer. Indian or Native Alaskan', 'Amer. Indian or Alaska Native','American Indian or Alaska Native', 'Nat. Hawaiin/Pac. Islander', 'Native Hawaiian/Other Pacific Islander', 'Hispanic or Latino','Hispanic/Latino'], [1,0,0,0,0,0,0,0,0,0,0], inplace=True)

  # Change DOB to age
  dob = data['udateofbirth'].tolist()
  agelist = []
  for x in dob:
    day = x[0:2]
    month = datetime.datetime.strptime(str(x[2:5]), '%b').month
    year = x[5:9]
    # Code from:
    # https://stackoverflow.com/questions/2217488/age-from-birthdate-in-python
    age = date.today().year - int(year) - ((date.today().month, date.today().day) < (month, day))
    agelist.append(age)    
  se = pd.Series(agelist)
  data['age'] = se.values 
  data.drop(['udateofbirth'],axis=1,inplace=True)

  data.to_csv(train_filename)

def load_retailer_data():
  X = []
  y = []
  x_control = []

  #Uncomment this the first time you run the file and then comment it back out:
  #clean_retailer_data()
  
  data = pd.read_csv(train_filename,error_bad_lines=False) 
  data.drop('0',axis=1,inplace=True)
  headers = list(data)
  y = data['hired'].astype(str).tolist()
  x_control = {}
  x_control['urace_orig'] = data['urace_orig'].astype(str).tolist() 

  data.drop(['urace_orig','hired'],axis=1,inplace=True)
  X = data.values.astype(str).tolist()

  return np.asarray(X), np.asarray(y), x_control

  #train = data.sample(frac=train_percentage)
  #test = data.drop(train.index)
  
  #return headers, train.values.tolist(), test.values.tolist()

def clean():
  clean_retailer_data()

def test():
  headers,train,test = load_data()
  print("load data unpacks correctly? -- ", (len(headers) != 0 and len(train) != 0 and len(test) != 0))  

  gathered_types = []
  for i, header in enumerate(headers):
    if all( isinstance(row[i], str) for row in train + test ):
      gathered_types.append(str)
    elif all( isinstance(row[i], float) for row in train + test ):
      gathered_types.append(float)
    elif all( isinstance(row[i], int) for row in train + test ):
      gathered_types.append(int)
    else:
      gathered_types.append(header,False)

  print("load_data types are correct? -- ", gathered_types == correct_types)
  print("all headers get types? -- ", len(headers) == len(gathered_types))

if __name__=="__main__":
#  clean()
  test()
