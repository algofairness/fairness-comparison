import numpy as np
import sys
sys.path.append('/home/jnim/fairness/fairness-comparison/data')
import pandas as pd


"""
0 = woman
1 = man
1 = good credit
0 = bad credit
"""
#Evan says that he set the weird male/female category to just male/female, scaled numerical values to be between 0 and 1, and removed unordered categorical features, and one-hot encoded ordered categorical data. 

def prepare_german(mode):
	if mode == 'numeric':
		X = []
		y = []
		x_control = []
		#os.chdir('../..')
		for line in open('../../data/preprocessed/german/german_credit_data.csv'):
			line = line.strip()
			if line == "": continue # skip+ empty lines

			if line[0] == "s": continue # skip line of feature categories, in csv
			line = line.split(",")

			"""
			Get class label
			"""
			class_label = line[-1]
			y.append(class_label)


			"""
			Repair sex variable
			"""

			if line[8] == "A95" or line[8] == "A92":
				line[8] = 0
			else:
				line[8] = 1


			line = line[:8]+line[9:-1]
			x_val = []

			"""
			Hot encode categorical data
			"""
			for i in range(0, len(line)):
				
				#I want x_val to be extended by each value in i after hot-encoding - set x_val = pd.get_dummies(protected)
				#Attribute 1float
				if i == 0:
					if line[i] == "A11":
						x_val.extend([1, 0, 0, 0])
					if line[i] == "A12":
						x_val.extend([0, 1, 0, 0])
					if line[i] == "A13":
						x_val.extend([0, 0, 1, 0])
					if line[i] == "A14":
						x_val.extend([0, 0, 0, 1])

				#Attribute 2
				if i == 1:
					x_val.append(line[i])

				#Attribute 3
				if i == 2:
					if line[i] == "A30":
						x_val.extend([1, 0, 0, 0, 0])
					if line[i] == "A31":
						x_val.extend([0, 1, 0, 0, 0])
					if line[i] == "A32":
						x_val.extend([0, 0, 1, 0, 0])
					if line[i] == "A33":
						x_val.extend([0, 0, 0, 1, 0])
					if line[i] == "A34":
						x_val.extend([0, 0, 0, 0, 1])
		
				#Attribute 5
				if i == 4:
					x_val.append(line[i])
		
				#Attribute 6
				if i == 5:
					if line[i] == "A61":
						x_val.extend([1, 0, 0, 0, 0])
					if line[i] == "A62":
						x_val.extend([0, 1, 0, 0, 0])
					if line[i] == "A63":
						x_val.extend([0, 0, 1, 0, 0])
					if line[i] == "A64":
						x_val.extend([0, 0, 0, 1, 0])
					if line[i] == "A65":
						x_val.extend([0, 0, 0, 0, 1])
		
				#Attribute 7
				if i == 6:
					if line[i] == "A71":
						x_val.extend([1, 0, 0, 0, 0])
					if line[i] == "A72":
						x_val.extend([0, 1, 0, 0, 0])
					if line[i] == "A73":
						x_val.extend([0, 0, 1, 0, 0])
					if line[i] == "A74":
						x_val.extend([0, 0, 0, 1, 0])
					if line[i] == "A75":
						x_val.extend([0, 0, 0, 0, 1])
		
				#Attribute 8
				if i == 7:
					x_val.append(line[i])
		
				#Attribute 10
				if i == 8:
					if line[i] == "A101":
						x_val.extend([1, 0, 0])
					if line[i] == "A102":
						x_val.extend([0, 1, 0])
					if line[i] == "A103":
						x_val.extend([0, 0, 1])
		
				#Attribute 13
				if i == 11:
					x_val.append(line[i])
		
				#Attributr 16
				if i == 14:
					x_val.append(line[i])
		
				#Attribute 17
				if i == 15:
					if line[i] == "A171":
						x_val.extend([1, 0, 0, 0])
					if line[i] == "A172":
						x_val.extend([0, 1, 0, 0])
					if line[i] == "A173":
						x_val.extend([0, 0, 1, 0])
					if line[i] == "A174":
						x_val.extend([0, 0, 0, 1])
		
				#Attribute 18
				if i == 16:
					x_val.append(line[i])


			X.append(x_val)

		x_control = {"sex": x_control}
		X = np.array(X)
		X = X.astype(float)
		y = np.array(y)
		y = y.astype(float)
		print(X)
		DataCSV = pd.DataFrame(X)
		#DataCSV.to_csv('Test.csv')
		return X, y, x_control
		print(X)

if __name__ == '__main__':
	prepare_german('numeric')

#print(load_german_data("german_credit_data.csv"))
