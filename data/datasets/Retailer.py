import pandas as pd
from Data import Data

class Retailer(Data):

    def __init__(self):
        Data.__init__(self)

        ## TODO: replace the below information with the correct info for your dataset.
        self.dataset_name = 'retailer'
        self.sensitive_attrs = ['urace_orig']
        self.unprotected_class_names = ['White']  ## TODO: check this
        self.categorical_features = []   ## TODO
        self.features_to_keep = [ 'usite', 'azip', 'urace_orig', 'udateofbirth',
                                  'ugender', 'szip', 'csvr2', 'hired' ]

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

    def get_dataset_name(self):
        return self.dataset_name

    def data_specific_processing(self, dataframe):
        ## TODO: any dataset sepcific preprocessing - this should include any ordered categorical
        ## replacement by numbers.
        return dataframe
