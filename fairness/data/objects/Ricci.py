import pandas as pd
from fairness.data.objects.Data import Data

class Ricci(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'ricci'
        # Class attribute will not be created until data_specific_processing is run.
        self.class_attr = 'Class'
        self.positive_class_val = 1
        self.sensitive_attrs = ['Race'] 
        self.privileged_class_names = ['W']
        self.categorical_features = [ 'Position' ]
        self.features_to_keep = [ 'Position', 'Oral', 'Written', 'Race', 'Combine' ]
        self.missing_val_indicators = []

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
