import sys
#from PreprocessHelpers.GermanProcessing import *
#from PreprocessHelpers.RicciCleaning import *
#from PreprocessHelpers.RetailerCleaning import *
import os
import pandas as pd
import fire

def prepareData(dataset=None):
    os.chdir('../data')
    if dataset == 'german' or dataset == None:
        print('German not done yet - have to redo according to paper')
	###################
	###GERMAN NOT DONE
	###################
	german_data = pd.read_csv(path + '/raw/german/german_data', sep = ' ', encoding = 'ISO-8859-1')
	
	#It's time to hard-code in the rows
	german_data.columns = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', \
	'savings', 'employment', 'investment_as_income_percentage', 'personal_status', \
	'other_debtors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', \
	'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', \
	'credit']
	
	german_data.to_csv(path + '/preprocessed/german_credit_data.csv', index = False) #Save the CSV because it gets used here and there
	
	#Load in the numeric data
	#german_data_numeric = pd.read_csv(path + '/raw/german/german_data_numeric', sep = ' ', encoding = 'ISO-8859-1')
	#
	#german_data.to_csv(path + '/preprocessed/german_credit_data.csv', index = False) #Save this CSV
	#print(len(german_data))
		
    if dataset == 'adult' or dataset == None: 
      	print('Adult not done yet - have to do according to paper')
	adult_data = pd.read_csv(path + 'adult/adult.data', sep = ' ', encoding = 'ISO-8859-1')
		
    if dataset == 'ricci' or dataset == None:
	ricci_Path = 'preprocessed/ricci/'
		
	ricci_Cleaned = clean_ricci_data()
	ricci_Cleaned.to_csv(ricci_Path + 'ricciCleaned.csv', index = False)
	#_data = pd.read_csv(path + '/raw/ricci/RicciDataMod.csv')
		
	
    if dataset == 'retailer' or dataset == None:
	retailer_Path = 'preprocessed/retailer/'
		
	retailer_Cleaned = clean_retailer_data()
	retailer_Cleaned.to_csv(retailer_Path + 'retailerCleaned.csv', index = False) 
	#retailer_data = pd.read_csv(path + '/raw/retailer/cleaned-apps-public.csv')
		
		
def main():
    fire.Fire(prepareData)
		
if __name__ == '__main__': 
    main()
