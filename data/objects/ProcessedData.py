import pandas as pd

TRAINING_PERCENT = 2.0 / 3.0

class ProcessedData():
    def __init__(self, data_obj):
        self.data = data_obj
        self.processed_df = pd.read_csv(self.data.get_processed_filename())
        self.numerical_df = pd.read_csv(self.data.get_processed_numerical_filename())
        self.binsensitive_df = pd.read_csv(self.data.get_processed_binsensitive_filename())
        self.processed_splits = []
        self.numerical_splits = []
        self.binsensitive_splits = []

    def get_processed_filename(self):
        return self.data.get_processed_filename()

    def get_processed_numerical_filename(self):
        return self.data.get_processed_numerical_filename()

    def get_processed_dataframe(self):
        return self.processed_df

    def get_processed_numerical_dataframe(self):
        return self.numerical_df

    def create_train_test_splits(self, num):
        if len(self.processed_splits) > 0:
            return self.processed_splits, self.numerical_splits

        for i in range(0, num):
            train = self.processed_df.sample(frac = TRAINING_PERCENT)
            test = self.processed_df.drop(train.index)
            self.processed_splits.append((train, test))
            train = self.numerical_df.sample(frac = TRAINING_PERCENT)
            test = self.numerical_df.drop(train.index)
            self.numerical_splits.append((train, test))
            train = self.binsensitive_df.sample(frac = TRAINING_PERCENT)
            test = self.binsensitive_df.drop(train.index)
            self.binsensitive_splits.append((train, test))

        return self.processed_splits, self.numerical_splits, self.binsensitive_splits

    def get_combined_sensitive_attr_name(self):
        return '-'.join(self.data.get_sensitive_attributes())
