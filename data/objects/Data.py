RAW_DATA_DIR = 'raw/'
PROCESSED_DATA_DIR = 'preprocessed/'
PROCESSED_ALL_STUB = "_processed.csv"
PROCESSED_NUM_STUB = "_numerical.csv"

class Data():
    def __init__(self):
        pass
	
    def get_dataset_name(self):
        """
        This is the stub name that will be used to generate the processed filenames and is the
        assumed stub for the raw data filename.
        """
        raise NotImplementedError("get_dataset_name() in Data is not implemented")

    def get_sensitive_attributes(self):
        """
        Returns a list of the names of any sensitive / protected attribute(s) that will be used 
        for a fairness analysis and should not be used to train the model.
        """
        raise NotImplementedError("get_sensitive_attributes() in Data is not implemented")

    def get_unprotected_class_names(self):
        raise NotImplementedError("get_unprotected_class_names() in Data is not implemented")

    def get_categorical_features(self):
        """
        Returns a list of features that should be expanded to one-hot versions for 
        numerical-only algorithms.  This should not include the protected features 
        or the outcome class variable.
        """
        raise NotImplementedError("get_categorical_features() in Data is not implemented")

    def get_features_to_keep(self):
        raise NotImplementedError("get_features_to_keep() in Data is not implemented")

    def get_missing_val_indicators(self):
        raise NotImplementedError("get_missing_val_indicators() in Data is not implemented")

    def data_specific_processing(self, dataframe):
        raise NotImplementedError("data_specific_processing() in Data is not implemented")

    def handle_missing_data(self, dataframe):
        """
        This method implements any data specific missing data processing.  Any missing data
        not replaced by values in this step will be removed by the general preprocessing 
        script.
        """
        raise NotImplementedError("handle_missing_data() in Data is not implemented")

    def get_raw_filename(self):
        return RAW_DATA_DIR + self.get_dataset_name() + '.csv'

    def get_processed_filename(self):
        return PROCESSED_DATA_DIR + self.get_dataset_name() + PROCESSED_ALL_STUB

    def get_processed_numerical_filename(self):
        return PROCESSED_DATA_DIR + self.get_dataset_name() + PROCESSED_NUM_STUB
