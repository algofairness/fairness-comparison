import pandas as pd
from data.objects.Data import Data

class Stub(Data):
    """
    Stub fake dataset that can be copied / used as an example for adding more dataset objects.
    """

    def __init__(self):
        Data.__init__(self)

        ## TODO: replace the below information with the correct info for your dataset.
        self.dataset_name = 'stub'
        self.sensitive_attrs = ['Race', 'Sex'] 
        self.unprotected_class_names = ['W', 'M']
        self.categorical_features = [ 'feature_name1', 'feature_name2' ]
        self.features_to_keep = [ 'feature_name1', 'feature_name2', 'feature_name3']
        self.missing_val_indicators = ['?']

    def get_dataset_name(self):
        return self.dataset_name

    def get_sensitive_attributes(self):
        """
        Returns a list of the names of any sensitive / protected attribute(s) that will be used 
        for a fairness analysis and should not be used to train the model.
        """
        return self.sensitive_attrs

    def get_unprotected_class_names(self):
        return self.unprotected_class_names

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

    def data_specific_processing(self, dataframe):
        ## TODO: any dataset sepcific preprocessing - this should include any ordered categorical
        ## replacement by numbers.
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
