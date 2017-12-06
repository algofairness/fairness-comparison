import sys
import os
import pandas as pd
import fire
from datasets.list import DATASETS

RAW_DATA_DIR = 'raw/'
PROCESSED_DATA_DIR = 'preprocessed/'
PROCESSED_ALL_STUB = "_processed.csv"
PROCESSED_NUM_STUB = "_numerical.csv"

def prepare_data():

    for dataset in DATASETS:
        data_path = RAW_DATA_DIR + dataset.get_dataset_name() + '.csv'
        data_frame = pd.read_csv(data_path)
        processed_data, processed_numerical = preprocess(dataset, data_frame)
        processed_file_name = PROCESSED_DATA_DIR + dataset.get_dataset_name() + PROCESSED_ALL_STUB
        print("Writing data to: " + processed_file_name)
        processed_data.to_csv(processed_file_name, index = False)
        numerical_file_name = PROCESSED_DATA_DIR + dataset.get_dataset_name() + PROCESSED_NUM_STUB
        print("Writing data to: " + numerical_file_name)
        processed_numerical.to_csv(numerical_file_name, index = False)

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
