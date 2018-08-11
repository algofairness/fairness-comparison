import math
import pandas as pd

from fairness.data.objects.Data import Data

class Sample(Data):
    """
    A way to sample from a dataset for testing purposes.
    num: the number of total items to sample (uniform with replacement)
    prob_pos_class: the probability [0,1] that an item has a positive class value
    prob_privileged: the probability [0,1] that an item has the privileged sensitive value (takes a
    binary view of sensitive attributes).  Requires that sensitive_attr is also set.
    sensitive_attr: the sensitive attribute to sample for the given probability privileged
    """
    def __init__(self, data, num = 100, prob_pos_class = "default",
                 prob_privileged = "default", sensitive_attr = "default"):
        Data.__init__(self)
        self.data = data
        self.dataset_name = data.get_dataset_name()
        self.class_attr = data.get_class_attribute()
        self.positive_class_val = data.get_positive_class_val("") # blank tag value
        self.sensitive_attrs = data.get_sensitive_attributes()
        self.privileged_class_names = data.get_privileged_class_names("") # blank tag value
        self.categorical_features = data.get_categorical_features()
        self.features_to_keep = data.get_features_to_keep()
        self.missing_val_indicators = data.get_missing_val_indicators()
        self.num_to_sample = num
        self.prob_pos_class = prob_pos_class
        self.prob_privileged = prob_privileged
        self.sensitive_attr = sensitive_attr
        if self.sensitive_attr == "default" and self.prob_privileged != "default":
            print("Error: using prob_privileged requires setting the sensitive_attr")
            exit(-1)
        for sens, priv in zip(self.sensitive_attrs, self.privileged_class_names):
            if sens == self.sensitive_attr:
                self.privileged_val = priv
                break

    def data_specific_processing(self, dataframe):
        dataframe = self.data.data_specific_processing(dataframe)
        return self.sample_prob_priv(dataframe, self.num_to_sample)

    def sample_number(self, dataframe, num):
        return dataframe.sample(n = num, replace=True)

    def sample_prob_pos(self, dataframe, num):
        if self.prob_pos_class != "default":
            num_pos = math.floor(self.prob_pos_class * num)
            num_neg = math.ceil((1 - self.prob_pos_class) * num)
            print("Sampling pos:" + str(num_pos) + " neg:" + str(num_neg))
            dataframe_pos = dataframe[dataframe[self.class_attr] == self.positive_class_val]
            dataframe_pos = self.sample_number(dataframe_pos, num_pos)
            dataframe_neg = dataframe[dataframe[self.class_attr] != self.positive_class_val]
            dataframe_neg = self.sample_number(dataframe_neg, num_neg)
            return pd.concat([dataframe_pos, dataframe_neg])
        else:
            return self.sample_number(dataframe, num)

    def sample_prob_priv(self, dataframe, num):
        if self.prob_privileged != "default":
            num_priv = math.floor(self.prob_privileged * num)
            num_unpriv = math.ceil((1 - self.prob_privileged) * num)
            print("Sampling privileged:" + str(num_priv) + " unprivileged:" + str(num_unpriv))
            dataframe_priv = dataframe[dataframe[self.sensitive_attr] == self.privileged_val]
            dataframe_priv = self.sample_prob_pos(dataframe_priv, num_priv)
            dataframe_unpriv = dataframe[dataframe[self.sensitive_attr] != self.privileged_val]
            dataframe_unpriv = self.sample_prob_pos(dataframe_unpriv, num_unpriv)
            return pd.concat([dataframe_priv, dataframe_unpriv])
        else:
            return self.sample_prob_pos(dataframe, num)

