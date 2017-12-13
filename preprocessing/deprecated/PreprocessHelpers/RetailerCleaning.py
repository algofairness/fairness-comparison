import datetime
from datetime import date
import time
import pandas as pd
import numpy as np
import csv
import random
import os

def clean_retailer_data():
	datadir = '../data'
	f = pd.read_csv(datadir + '/raw/retailer/cleaned-apps-public.csv',error_bad_lines=False, encoding = 'ISO-8859-1')
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
		day = int(x[0:2])
		month = int(datetime.datetime.strptime(str(x[2:5]), '%b').month)
		year = int(x[5:9])
		# Code from:
		# https://stackoverflow.com/questions/2217488/age-from-birthdate-in-python
		age = date.today().year - int(year) - ((date.today().month, date.today().day) < (month, day))
		agelist.append(age)    
	se = pd.Series(agelist)
	data['age'] = se.values 
	data.drop(['udateofbirth'],axis=1,inplace=True)
	
	return data
	
