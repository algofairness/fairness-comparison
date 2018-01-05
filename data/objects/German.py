import pandas as pd
from data.objects.Data import Data

class German(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'german'
        self.class_attr = 'credit'
        self.sensitive_attrs = ['sex']
        self.unprotected_class_names = ['male']
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

    def get_class_attribute(self):
        """
        Returns the name of the class attribute to be used for classification.
        """
        return self.class_attr

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
        sexdict = {('A91', 'A93', 'A94'): 'male', ('A92', 'A95'):'female'}
        dataframe['sex'] = dataframe['personal_status'].map(sexdict)
        dataframe.drop('personal_status', 1)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
