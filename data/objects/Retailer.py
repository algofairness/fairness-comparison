import datetime
from datetime import date
import pandas as pd

from data.objects.Data import Data

class Retailer(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'retailer'
        self.class_attr = 'hired' 
        self.positive_class_val = '1'
        self.sensitive_attrs = ['urace_orig']
        self.privileged_class_names = ['White']
        self.categorical_features = []   ## TODO
        self.features_to_keep = [ 'usite', 'azip', 'urace_orig', 'udateofbirth',
                                  'ugender', 'szip', 'csvr2', 'hired' ]
        self.missing_val_indicators = ['""']

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
        dataframe = dataframe.assign(age = se.values)
        dataframe.drop(['udateofbirth'], axis=1, inplace=True)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
