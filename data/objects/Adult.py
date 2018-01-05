from data.objects.Data import Data
	
class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'income-per-year'
        self.sensitive_attrs = ['race', 'sex']
        self.unprotected_class_names = ['White', 'Male']
        self.categorical_features = [ 'workclass', 'education', 'marital-status', 'occupation', 
                                      'relationship', 'native-country' ]
        self.features_to_keep = [ 'age', 'workclass', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                  'capital-loss', 'hours-per-week', 'native-country',
                                  'income-per-year' ]
        self.missing_val_indicators = ['?']
	
    def get_dataset_name(self):
        """
        This is the stub name that will be used to generate the processed filenames and is the
        assumed stub for the raw data filename.
        """
        return "adult"

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

    def get_missing_val_indicators(self):
        return self.missing_val_indicators

    def data_specific_processing(self, dataframe):
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe
