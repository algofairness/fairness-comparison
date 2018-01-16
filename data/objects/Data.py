RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/preprocessed/'
RESULT_DIR = "results/"
ANALYSIS_DIR = "analysis/"

PROCESSED_ALL_STUB = "_processed.csv"
RESULTS_ALL_STUB = "_results.csv"
ANALYSIS_ALL_STUB = "_analysis.csv"
NUM_STUB = "_numerical.csv"
BINSENS_STUB = "_numerical_binsensitive.csv"

class Data():
    def __init__(self):
        pass

    def get_dataset_name(self):
        """
        This is the stub name that will be used to generate the processed filenames and is the
        assumed stub for the raw data filename.
        """
        return self.dataset_name

    def get_class_attribute(self):
        """
        Returns the name of the class attribute to be used for classification.
        """
        return self.class_attr

    def get_positive_class_val(self):
        """
        Returns the value used in the dataset to indicate the positive classification choice.
        """
        return self.positive_class_val

    def get_sensitive_attributes(self):
        """
        Returns a list of the names of any sensitive / protected attribute(s) that will be used
        for a fairness analysis and should not be used to train the model.
        """
        return self.sensitive_attrs

    def get_combined_sensitive_attr_name(self):
        return '-'.join(self.get_sensitive_attributes())

    def append_sensitive_attribute(self, attr):
        """
        Adds a new sensitive attribute to the end of the list of sensitive attributes.  Used to
        create a new sensitive attribute that combines all the other ones.
        """
        self.sensitive_attrs.append(attr)

    def get_privileged_class_names(self):
        """
        Returns a list in the same order as the sensitive attributes list above of the
        privileged class name (exactly as it appears in the data) of the associated sensitive
        attribute.
        """
        return self.privileged_class_names

    def get_categorical_features(self):
        """
        Returns a list of features that should be expanded to one-hot versions for
        numerical-only algorithms.  This should not include the protected features
        or the outcome class variable.
        """
        return self.categorical_features

    def get_features_to_keep(self):
        return self.features_to_keep

    def get_missing_val_indicators(self):
        return self.missing_val_indicators

    def get_raw_filename(self):
        return RAW_DATA_DIR + self.get_dataset_name() + '.csv'

    def get_processed_filename(self):
        return PROCESSED_DATA_DIR + self.get_dataset_name() + PROCESSED_ALL_STUB

    def get_processed_numerical_filename(self):
        return PROCESSED_DATA_DIR + self.get_dataset_name() + NUM_STUB

    def get_processed_binsensitive_filename(self):
        """
        Returns the filename for a processed version of this dataset that is all numerical with a
        binary (numerical) sensitive attribute.  All privileged values will be replaced by 1 and
        all other sensitive values by 0.
        """
        return PROCESSED_DATA_DIR + self.get_dataset_name() + BINSENS_STUB

    def get_results_filename(self, sensitive_attr):
        return RESULT_DIR + self.get_dataset_name() + "_" + sensitive_attr + RESULTS_ALL_STUB

    def get_results_numerical_filename(self, sensitive_attr):
        return RESULT_DIR + self.get_dataset_name() + "_" + sensitive_attr + NUM_STUB

    def get_results_numerical_binsensitive_filename(self, sensitive_attr):
        return RESULT_DIR + self.get_dataset_name() + "_" + sensitive_attr + BINSENS_STUB

    def get_analysis_filename(self, sensitive_attr):
        return ANALYSIS_DIR + self.get_dataset_name() + "_" + sensitive_attr + ANALYSIS_ALL_STUB

    def get_analysis_numerical_filename(self, sensitive_attr):
        return ANALYSIS_DIR + self.get_dataset_name() + "_" + sensitive_attr + NUM_STUB

    def get_analysis_numerical_binsensitive_filename(self, sensitive_attr):
        return ANALYSIS_DIR + self.get_dataset_name() + "_" + sensitive_attr + BINSENS_STUB

    def data_specific_processing(self, dataframe):
        """
        Takes a pandas dataframe and modifies it to do any data specific processing.  This should
        include any ordered categorical replacement by numbers.  The resulting pandas dataframe is
        returned.
        """
        return dataframe

    def handle_missing_data(self, dataframe):
        """
        This method implements any data specific missing data processing.  Any missing data
        not replaced by values in this step will be removed by the general preprocessing
        script.
        """
        return dataframe

