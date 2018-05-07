from data.objects.Data import Data

class Sample(Data):
    """
    A way to sample from a dataset for testing purposes.
    """
    def __init__(self, data, num = 100):
        self.data = data
        self.dataset_name = data.get_dataset_name()
        self.class_attr = data.get_class_attribute()
        self.positive_class_val = data.get_positive_class_val("") # sigh
        self.sensitive_attrs = data.get_sensitive_attributes()
        self.privileged_class_names = data.get_privileged_class_names("") # sigh
        self.categorical_features = data.get_categorical_features()
        self.features_to_keep = data.get_features_to_keep()
        self.missing_val_indicators = data.get_missing_val_indicators()
        self.num_to_sample = num

    def data_specific_processing(self, dataframe):
        dataframe = self.data.data_specific_processing(dataframe)
        return dataframe.sample(n = self.num_to_sample, replace=True)
