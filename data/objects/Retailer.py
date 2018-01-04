import datetime
from datetime import date
import pandas as pd

from data.objects.Data import Data

class Retailer(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'retailer'
        self.class_attr = 'hired'
        self.sensitive_attrs = ['urace_orig']
        self.unprotected_class_names = ['White']
        self.categorical_features = []   ## TODO
        self.features_to_keep = [ 'usite', 'azip', 'urace_orig', 'udateofbirth',
                                  'ugender', 'szip', 'csvr2', 'hired' ]
        self.missing_val_indicators = ['""']

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
        # Change DOB to age
        dob = dataframe['udateofbirth'].tolist()
        agelist = []
        for x in dob:
            day = x[0:2]
            month = datetime.datetime.strptime(str(x[2:5]), '%b').month
            year = x[5:9]
            # Code from:
            # https://stackoverflow.com/questions/2217488/age-from-birthdate-in-python
            # TODO: this should be the age when the hiring decision was made, not the age today
            age = date.today().year - int(year) - ((date.today().month, date.today().day) < (month, int(day)))
            agelist.append(age)    
        se = pd.Series(agelist)
        dataframe['age'] = se.values 
        dataframe.drop(['udateofbirth'], axis=1, inplace=True)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
