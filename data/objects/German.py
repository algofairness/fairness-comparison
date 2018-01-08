import pandas as pd
from data.objects.Data import Data

class German(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'german'
        self.class_attr = 'credit'
        self.positive_class_val = '1' 
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = ['male']
        self.categorical_features = ['status', 'credit_history', 'purpose', 'savings', 'employment', 
                                     'personal_status', 'other_debtors', 'property', 'installment_plans', 
                                     'housing', 'skill_level', 'telephone', 'foreign_worker'] 
        self.features_to_keep = [ 'status', 'month', 'credit_history', 'purpose', 'credit_amount',
                                  'savings', 'employment', 'investment_as_income_percentage', 
                                  'personal_status', 'other_debtors', 'residence_since',
                                  'property', 'age', 'installment_plans', 'housing', 
                                  'number_of_credits', 'skill_level', 'people_liable_for',
                                  'telephone', 'foreign_worker', 'credit' ]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        sexdict = {('A91', 'A93', 'A94'): 'male', ('A92', 'A95'):'female'}
        dataframe['sex'] = dataframe['personal_status'].map(sexdict)
        dataframe.drop('personal_status', 1)

        ## TODO: convert age as done in calders so that it's a binary sensitive attribute
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
