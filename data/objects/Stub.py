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
        self.privileged_class_names = ['W', 'M']
        self.class_attr = 'class_feature_name'
        self.positive_class_val = 'yes'
        self.categorical_features = [ 'feature_name1', 'feature_name2' ]
        self.features_to_keep = [ 'feature_name1', 'feature_name2', 'feature_name3']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        ## TODO: any dataset sepcific preprocessing - this should include any ordered categorical
        ## replacement by numbers.
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
