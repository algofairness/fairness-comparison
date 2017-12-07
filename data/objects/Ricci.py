import pandas as pd
from Data import Data

class Ricci(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'ricci'
        self.sensitive_attrs = ['Race'] 
        self.unprotected_class_names = ['W']
        self.categorical_features = [ 'Position' ]
        self.features_to_keep = [ 'Position', 'Oral', 'Written', 'Race', 'Combine' ]
        self.missing_val_indicators = []

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

    def get_missing_val_indicators(self):
        return self.missing_val_indicators

    def data_specific_processing(self, dataframe):
        dataframe['Class'] = dataframe.apply(passing_grade, axis=1)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe

def passing_grade(row):
    """
    A passing grade in the Ricci data is defined as any grade above a 70 in the combined
    oral and written score.  (See Miao 2010.)
    """
    if row['Combine'] >= 70.0:
        return 1
    else:
        return 0
