import sys
from dataset_info import *
import os
import pandas as pd
import fire

def prepare_data(dataset_names=DATASETS):

    for dataset in dataset_names:
        data_path = RAW_DATA_DIR + dataset + '.csv'
        data_frame = pd.read_csv(data_path)
        processed_data, processed_numerical = preprocess(dataset, data_frame)
        processed_data.to_csv(PROCESSED_DATA_DIR + dataset + '.csv')
        processed_numerical.to_csv(PROCESSED_DATA_DIR + dataset + '_numerical.csv')

def preprocess(dataset_name, data_frame):
    """
    The preprocess function takes a pandas data frame and returns two modified data frames:
    1) all the data as given with any features that should not be used for training or fairness
    analysis removed.
    2) only the numerical and ordered categorical data, sensitive attributes, and class attribute.
    Categorical attributes are one-hot encoded.
    """
    data_frame = data_frame[FEATURES_TO_KEEP[dataset_name]].dropna()
    ## TODO: any dataset sepcific preprocessing
    ## TODO: replace non-protected with single value as needed
    ## TODO: create two versions of the data
    return data_frame, data_frame	
		
def main():
    fire.Fire(prepare_data)
		
if __name__ == '__main__': 
    main()
