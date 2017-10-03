# README

## This file is for running all benchmarks on one or all of the datasets in the fairness-comparison github repository. 

The file has five inputs: Data, Missing, Features, Output, Protected.

**Data**: 
The Data keyword refers to the dataset we wish to run benchmarks on. All of the datasets avaiable have a folder in the 
__fairness-comparison/raw directory__, and their names are simply the names of their corresponding csv in the raw folder 
without the ".csv". For example, to run benchmarks on the German Credit Dataset, one would enter "German" for the Data
keyword. 

**Missing**:
The Missing keyword refers to how we wish to treat the missing data. The only option as of now is to simply drop the
missing data. This is done by entering "Drop" for the Missing keyword.

**Features**: 
This Keyword is for telling us how we should treat our features. The only option for now is to treat the entire column as 
numerical. This is done by entering "Numerical" for the Features Keyword.

**Output**: 
This Keyword is for naming the output file containing the analysis. Output files will be in the __fairness-comparison/results__
directory. 



