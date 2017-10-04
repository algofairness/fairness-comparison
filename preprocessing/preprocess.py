import sys
sys.path.append('/home/h205c/jnim/fairness/fairness-comparison/data')
import pandas as pd
def prepareData():
	path = '../data/'
	if dataset == 'german':
		
		german_data = pd.read_csv(path + 'german/german_data', sep = ' ', encoding = 'ISO-8859-1')
		
		#It's time to hard-code in the rows
		german_data.columns = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', \
		'savings', 'employment', 'investment_as_income_percentage', 'personal_status', \
		'other_debtors', 'residence_since', 'property', 'age', 'installment_plans', 'housing', \
		'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', \
		'credit']
		
		german_data.to_csv(path + '/preprocessed/german_credit_data.csv', index = False)
		
		print(len(german))
		
	if dataset == 'adult': 
		adult_data = pd.read_csv(path + 'adult/adult.data', sep
		
		
if __name__ == '__main__': 
	prepareData()
