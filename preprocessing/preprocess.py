import sys
from PreprocessHelpers.GermanProcessing import *
from PreprocessHelpers.RicciCleaning import *
from PreprocessHelpers.RetailerCleaning import *
import os
import pandas as pd

def prepareData(dataset, datatype):
	os.chdir('../data')
	if dataset == 'german':
		print('German not done yet - have to redo according to paper')
		###################
		###GERMAN NOT DONE
		###################
		if datatype == 'categorical':
			path = 'preprocessed/German'
			german_data = pd.read_csv('/raw/german/german_data', sep = ' ', encoding = 'ISO-8859-1')
			
			#It's time to hard-code in the rows
			german_data.columns = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', \
			'savings', 'employment', 'investment_as_income_percentage', 'personal_status', \
			'other_debtors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', \
			'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', \
			'credit']
		
			german_data.to_csv('/preprocessed/german_credit_data.csv', index = False) #Save the CSV because it gets used here and there
		
		if datatype == 'numeric':
			#Load in the numeric data - first create array where we keep data 
			newlines = []
			i = 0
			#while i< 25:
			#	newlines.append([])
			#	i +=1
				
			for line in open('raw/german/german.data-numeric'):
				#if "  " in line: #If there are multiple spaces, we want to replace them with one space so that the data can be read in.
				newline = line.split()
				newlines.append(newline)
				
			print(newlines[0])
			CSVHeaders = ['One',	'Two',	'three',	'four',	'Five',	'Six',	'Seven', \
			'eight', 'nine', 'Ten', 'Eleven', 'Twelve', '13', '14', '15', '16', '17',	'18', '19',\
			'20',	'21',	'22',	'23',	'gender', 'Credit']			
			german_data_numeric = pd.DataFrame(newlines, columns = CSVHeaders)
				
			german_data_numeric.to_csv('preprocessed/german/german_numeric.csv', index = False)

		
	if dataset == 'adult': 
		print('Adult not done yet - have to do according to paper')
		adult_data = pd.read_csv(path + 'adult/adult.data', sep = ' ', encoding = 'ISO-8859-1')
		
	if dataset == 'ricci':
		ricci_Path = 'preprocessed/ricci/'
		
		ricci_Cleaned = clean_ricci_data()
		ricci_Cleaned.to_csv(ricci_Path + 'ricciCleaned.csv', index = False)
		#_data = pd.read_csv(path + '/raw/ricci/RicciDataMod.csv')
		
	
	if dataset == 'retailer':
		retailer_Path = 'preprocessed/retailer/'
		
		retailer_Cleaned = clean_retailer_data()
		retailer_Cleaned.to_csv(retailer_Path + 'retailerCleaned.csv', index = False) 
		#retailer_data = pd.read_csv(path + '/raw/retailer/cleaned-apps-public.csv')
		
		
		
if __name__ == '__main__': 
	prepareData('german', 'numeric')
