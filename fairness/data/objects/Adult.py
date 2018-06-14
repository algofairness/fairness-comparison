from fairness.data.objects.Data import Data
	
class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'income-per-year'
        self.positive_class_val = '>50K'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'Male']
        self.categorical_features = [ 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country' ]
        self.features_to_keep = [ 'age', 'workclass', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                  'capital-loss', 'hours-per-week', 'native-country',
                                  'income-per-year' ]
        self.missing_val_indicators = ['?']
