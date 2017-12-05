import sys
from preprocessing.preprocess import *
from data.dataset_info import *
from algorithms.algorithm_info import *

import fire

from preprocessing.prepare_german import *
from algorithms.gen.GenAlgorithm import *

from run_metrics import *

#import argparse
#
#Create parser to process inputs
#parser = argparse.ArgumentParser()
#parser.add_argument('--algorithm', action='store', default = 'all', dest='algorithm',
#                    help='Choose the algorithm we will use to analyze the data. Available algorithms are in the "fairness-comparison/algorithms directory". Entering "all" will run all inputs'. )
#
#parser.add_argument('--data', action='store', default = 'German', dest='dataset',
#                    help='Choose dataset. All datasets have a folder in the fairness-comparison/raw directory.')
#
#parser.add_argument('--missing', action='store', default = 'drop', dest='missing',
#                    help='Choose method of dealing with missing values. Only option is drop all missing.')
#
#parser.add_argument('--features', action='store', default = 'numerical_only', dest='features',
#                    help='Choose which features to include in analysis. Options are numerical_only and categorical_only.')
#
#parser.add_argument('--output', action='store', default = 'German_Test', dest='output',
#                    help='Name of output file. File will be outputted to the fairness-comparison/results directory.')
#
#parser.add_argument('--protected', action='store', default = 'race', dest='protected_feature',
#                    help='Choose which attribute(s) will not be analyzed. Multiple attributes can be entered so long as they are entered in one string separated by commas (e.g: "race,gender")')
#
#inputs = parser.parse_args()
#
##clean data
#
#if inputs.algorithm == 'all':
	run_metrics('german', prepare_german, classify_german)


def benchmark(algorithm = ALGORITHMS, dataset_names = DATASETS):


#General idea: Run gen_algo with inputs since Derek has done most of my work for me

print results.dataset

def main():
    fire.Fire(benchmark)
		
if __name__ == '__main__': 
    main()
