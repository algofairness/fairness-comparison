import sys
import os
import pandas as pd
import fire
from data.objects.list import DATASETS, get_dataset_names

def prepare_data(dataset_names = get_dataset_names()):

    for dataset in DATASETS:
        if not dataset.get_dataset_name() in dataset_names:
            continue
        print("--- Processing dataset:" + dataset.get_dataset_name() + " ---")
        data_path = dataset.get_raw_filename()
        ## TODO: right now the retailer data won't load without ignoring errors - fix this
        ## and remove the below error_bad_lines=False.
        data_frame = pd.read_csv(data_path, error_bad_lines=False, 
                                 na_values=dataset.get_missing_val_indicators())
	
        processed_data, processed_numerical = preprocess(dataset, data_frame)
	
        processed_file_name = dataset.get_processed_filename()
        print("Writing data to: " + processed_file_name)
        processed_data.to_csv(processed_file_name, index = False)

        numerical_file_name = dataset.get_processed_numerical_filename()
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

    # Remove any columns not included in the list of features to keep.
    smaller_data = data_frame[dataset.get_features_to_keep()]

    # Handle missing data.
    missing_processed = dataset.handle_missing_data(smaller_data) 

    # Remove any rows that have missing data.
    missing_data_removed = missing_processed.dropna()
    missing_data_count = missing_processed.shape[0] - missing_data_removed.shape[0]
    if missing_data_count > 0:
        print("Missing Data: " + str(missing_data_count) + " rows removed from dataset " + dataset.get_dataset_name()) 

    # Do any data specific processing.
    processed_data = dataset.data_specific_processing(missing_data_removed)

    # Create a one-hot encoding of the categorical variables.
    processed_numerical = pd.get_dummies(processed_data, 
                                         columns = dataset.get_categorical_features())

    ## TODO: replace non-protected with single value as needed

    return processed_data, processed_numerical	
		
def main():
    fire.Fire(prepare_data)
		
if __name__ == '__main__': 
    main()
