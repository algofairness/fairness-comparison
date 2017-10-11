import csv
import pandas as pd
import numpy as np

def clean_ricci_data():
	datadir = '../data'
	f = pd.read_csv(datadir + '/raw/ricci/RicciDataMod.csv',error_bad_lines=False, encoding = 'ISO-8859-1')
	f.Race.replace(['W','B','H'], [1,0,0], inplace=True)
	f.Position.replace(['Captain','Lieutenant'], [1,0], inplace=True)
	
	return f
