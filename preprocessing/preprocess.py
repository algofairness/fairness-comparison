import sys
from PreprocessHelpers.GermanPreprocessing import *
from PreprocessHelpers.RetailerCleaning import *
import os
import pandas as pd

def prepareData(dataset, datatype):
	os.chdir('../data')
	if dataset == 'german':
		path = 'preprocessed/german/'
		
		if datatype == 'categorical':
			#Load in categorical data to the array below
			newlines = []
		
			for line in open('raw/german/german_data'):
				newline = line.split() #Since the entries are separated by multiple spaces 
				newlines.append(newline)	
			#german_data = pd.read_csv('raw/german/german_data', sep = ' ', encoding = 'UTF-8')
			
			#It's time to hard-code in the header row
			german_data_categorical = pd.DataFrame(newlines)
			
			csvHeaders = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings', \
			'employment', 'investment_as_income_percentage', 'personal_status', 'other_debtors', \
			'residence_since', 'property', 'age', 'installment_plans', 'housing', 'number_of_credits',\
			'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', 'credit']

			german_data_categorical.columns = csvHeaders
			
			german_data_categorical.to_csv(path + 'german_credit_data.csv', index = False, header = True) 
	
		if datatype == 'numeric':
			#Load in the numeric data - first create array where we keep data 
			newlines = []
			
			for line in open('raw/german/german.data-numeric'):
				newline = line.split() #Since the entries are separated by multiple spaces 
				newlines.append(newline)
				
			print(newlines[0])
			csvHeaders = ['One', 'Two', 'three', 'four', 'Five', 'Six', 'gender', 'Seven', 'eight',	\
			'nine',	'Ten', 'Eleven', 'Twelve', '13', '14', '15', '16', '17', '18', '19', '20', '21', \
			'22', '23', 'Credit']

			german_data_numeric = pd.DataFrame(newlines, columns = csvHeaders)	
			
			
			orderedHeaders = ['One',	'Two',	'three',	'four',	'Five',	'Six',	'Seven', \
			'eight', 'nine', 'Ten', 'Eleven', 'Twelve', '13', '14', '15', '16', '17',	'18', '19',\
			'20',	'21',	'22',	'23',	'gender', 'Credit']			
			
			#Put columns in correct order
			german_data_numeric = german_data_numeric[orderedHeaders]
				
			german_data_numeric.to_csv(path + 'german_numeric.csv', index = False)

		
	if dataset == 'adult': 
		path = 'preprocessed/adult/'
		#adult_data = pd.read_csv(path + 'adult/adult.data', sep = ' ', encoding = 'ISO-8859-1')
		new_lines = []
		count = 0
		for line in open("raw/adult/adult.data"):
		    line = line.strip()
		    count +=1
		    if line == "": continue # skip empty lines
		    if line[0] == "a": continue # skip line of feature categories, in csv
		    line = line.split(",")
		    if len(line) != 15 or "?" in line: # if a line has missing attributes, ignore it
		        continue
		    else:
		        new_lines.append(line)
		
		
		f = open(path + "adult.csv", 'w')
		for i in new_lines:
		    """
		    Convert -1 to 0 for Kamashima's classifiers
		    """
		    x = ""
		    for j in i:
		        x += str(j)
		        x +=","
		    x = x[:-1]
		    f.write(x)
		    f.write('\n')
		f.close()
	
	if dataset == 'retailer':
		path = 'preprocessed/retailer/'
		
		retailer_Cleaned = clean_retailer_data()
		retailer_Cleaned.to_csv(path + 'retailerCleaned.csv', index = False) 
		
	if dataset == 'ricci':
		path = 'preprocessed/ricci/'
		f = pd.read_csv('raw/ricci/RicciDataMod.csv',error_bad_lines=False)
		f.Race.replace(['W','B','H'], [1,0,0], inplace=True)
		f.Position.replace(['Captain','Lieutenant'], [1,0], inplace=True)
		
		f.to_csv(path + 'ricci.csv', index=False)
		
		#Nothing to do because Ricci is done
		#_data = pd.read_csv(path + '/raw/ricci/RicciDataMod.csv')
		
	
	
		#retailer_data = pd.read_csv(path + '/raw/retailer/cleaned-apps-public.csv')
		
		
		
if __name__ == '__main__': 
	prepareData('ricci', 'categorical')
