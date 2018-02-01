import pandas as pd

TRAINING_PERCENT = 2.0 / 3.0

# FIXME the set of available tags should exist somewhere so this isn't hard-coded
TAGS = ["original", "numerical", "numerical-binsensitive"]

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
