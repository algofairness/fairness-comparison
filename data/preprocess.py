import sys
from dataset_info import *
import os
import pandas as pd
import fire

from Ricci import Ricci

DATASETS = [ Ricci() ]

RAW_DATA_DIR = 'raw/'
PROCESSED_DATA_DIR = 'preprocessed/'

def prepare_data(dataset_names=DATASETS):

    for dataset in DATASETS:
        data_path = RAW_DATA_DIR + dataset.get_dataset_name() + '.csv'
        data_frame = pd.read_csv(data_path)
        processed_data, processed_numerical = preprocess(dataset, data_frame)
        print("Writing data to: " + PROCESSED_DATA_DIR + dataset.get_dataset_name() + '.csv')
        processed_data.to_csv(PROCESSED_DATA_DIR + dataset.get_dataset_name() + '.csv', index = False)
        print("Writing data to: " + PROCESSED_DATA_DIR + dataset.get_dataset_name() + '_numerical.csv')
        processed_numerical.to_csv(PROCESSED_DATA_DIR + dataset.get_dataset_name() + '_numerical.csv', index = False)

def preprocess(dataset, data_frame):
    """
    The preprocess function takes a pandas data frame and returns two modified data frames:
    1) all the data as given with any features that should not be used for training or fairness
    analysis removed.
    2) only the numerical and ordered categorical data, sensitive attributes, and class attribute.
    Categorical attributes are one-hot encoded.
    """
    smaller_data = data_frame[dataset.get_features_to_keep()]
    processed_data = dataset.data_specific_processing(smaller_data)

    ## TODO: handle missing data, which may be indicated differently per data set.  If missing
    ## data should be treated as a category, then it needs to be replaced by a np.nan

    # Create a one-hot encoding of the categorical variables.
    processed_numerical = pd.get_dummies(processed_data, 
                                         columns = dataset.get_categorical_features(),
                                         dummy_na=True)

    ## TODO: replace non-protected with single value as needed

    return processed_data, processed_numerical	
		
def main():
    fire.Fire(prepare_data)
		
if __name__ == '__main__': 
    main()
