import pandas as pd
import pathlib
from fairness.results import local_results_path

BASE_DIR = local_results_path()
PACKAGE_DIR = pathlib.Path(__file__).parents[2]
RAW_DATA_DIR = PACKAGE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PACKAGE_DIR / 'data' / 'preprocessed'
RESULT_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "analysis"


class Data():
    def __init__(self):
        pass

    def get_dataset_name(self):
        """
        This is the stub name that will be used to generate the processed filenames and is the
        assumed stub for the raw data filename.
        """
        return self.dataset_name

    def get_class_attribute(self):
        """
        Returns the name of the class attribute to be used for classification.
        """
        return self.class_attr

    def get_positive_class_val(self, tag):
        """
        Returns the value used in the dataset to indicate the positive classification choice.
        """
        # FIXME this dependence between tags and metadata is bad; don't know how to fix it right now
        if tag == 'numerical-binsensitive':
            return 1
        else:
            return self.positive_class_val

    def get_sensitive_attributes(self):
        """
        Returns a list of the names of any sensitive / protected attribute(s) that will be used
        for a fairness analysis and should not be used to train the model.
        """
        return self.sensitive_attrs

    def get_sensitive_attributes_with_joint(self):
        """
        Same as get_sensitive_attributes, but also includes the joint sensitive attribute if there
        is more than one sensitive attribute.
        """
        if len(self.get_sensitive_attributes()) > 1:
            return self.get_sensitive_attributes() + ['-'.join(self.get_sensitive_attributes())]
        return self.get_sensitive_attributes()

    def get_privileged_class_names(self, tag):
        """
        Returns a list in the same order as the sensitive attributes list above of the
        privileged class name (exactly as it appears in the data) of the associated sensitive
        attribute.
        """
        # FIXME this dependence between tags and privileged class names is bad; don't know how to
        # fix it right now
        if tag == 'numerical-binsensitive':
            return [1 for x in self.get_sensitive_attributes()]
        else:
            return self.privileged_class_names

    def get_privileged_class_names_with_joint(self, tag):
        """
        Same as get_privileged_class_names, but also includes the joint sensitive attribute if there
        is more than one sensitive attribute.
        """
        priv_class_names = self.get_privileged_class_names(tag)
        if len(priv_class_names) > 1:
            return priv_class_names + ['-'.join(str(v) for v in priv_class_names)]
        return priv_class_names

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

    def load_raw_dataset(self):
        data_path = self.get_raw_filename()
        data_frame = pd.read_csv(data_path, error_bad_lines=False,
                                 na_values=self.get_missing_val_indicators(),
                                 encoding = 'ISO-8859-1')
        return data_frame

    def get_raw_filename(self):
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return RAW_DATA_DIR / (self.get_dataset_name() + '.csv')

    def get_filename(self, tag):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return PROCESSED_DATA_DIR / (self.get_dataset_name() + "_" + tag + '.csv')

    def get_results_filename(self, sensitive_attr, tag):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR / (self.get_dataset_name() + "_" + sensitive_attr + "_" + tag + '.csv')

    def get_param_results_filename(self, sensitive_attr, tag, algname):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR / (algname + '_' + self.get_dataset_name() + "_" + sensitive_attr + \
               "_" + tag + '.csv')

    def get_analysis_filename(self, sensitive_attr, tag):
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        return ANALYSIS_DIR / (self.get_dataset_name() + "_" + sensitive_attr + "_" + tag + '.csv')

    def data_specific_processing(self, dataframe):
        """
        Takes a pandas dataframe and modifies it to do any data specific processing.  This should
        include any ordered categorical replacement by numbers.  The resulting pandas dataframe is
        returned.
        """
        return dataframe

    def handle_missing_data(self, dataframe):
        """
        This method implements any data specific missing data processing.  Any missing data
        not replaced by values in this step will be removed by the general preprocessing
        script.
        """
        return dataframe

    def get_class_balance_statistics(self, data_frame=None):
        if data_frame is None:
            data_frame = self.load_raw_dataset()
        r = data_frame.groupby(self.get_class_attribute()).size()
        return r

    def get_sensitive_attribute_balance_statistics(self, data_frame=None):
        if data_frame is None:
            data_frame = self.load_raw_dataset()
        return [data_frame.groupby(a).size()
                for a in self.get_sensitive_attributes()]

    ##########################################################################

    def get_results_data_frame(self, sensitive_attr, tag):
        return pd.read_csv(self.get_results_filename(sensitive_attr, tag))

    def get_param_results_data_frame(self, sensitive_attr, tag):
        return pd.read_csv(self.get_param_results_filename(sensitive_attr, tag))
