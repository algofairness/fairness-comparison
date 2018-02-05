import pandas as pd

TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive"]
TRAINING_PERCENT = 2.0 / 3.0

class ProcessedData():
    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False

    def get_processed_filename(self, tag):
        return self.data.get_filename(tag)

    def get_dataframe(self, tag):
        return self.dfs[tag]

    def create_train_test_splits(self, num):
        if self.has_splits:
            return self.splits

        for i in range(0, num):
            for (k, v) in self.dfs.items():
                train = self.dfs[k].sample(frac = TRAINING_PERCENT)
                test = self.dfs[k].drop(train.index)
                self.splits[k].append((train, test))

        self.has_splits = True
        return self.splits

    def get_sensitive_values(self, tag):
        """
        Returns a dictionary mapping sensitive attributes in the data to a list of all possible
        sensitive values that appear.
        """
        df = self.get_dataframe(tag)
        all_sens = self.data.get_sensitive_attributes_with_joint()
        sensdict = {}
        for sens in all_sens:
             sensdict[sens] = list(set(df[sens].values.tolist()))
        print(sensdict)
        return sensdict

