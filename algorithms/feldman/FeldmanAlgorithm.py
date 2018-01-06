from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from pandas import DataFrame
from algorithms.Algorithm import Algorithm

REPAIR_LEVEL_DEFAULT = 1.0

class FeldmanAlgorithm(Algorithm):
    def __init__(self, algorithm, repair_level=REPAIR_LEVEL_DEFAULT):
        Algorithm.__init__(self)
        self.model = algorithm
        self.name = 'Feldman-' + self.model.get_name()
        self.repair_level = REPAIR_LEVEL_DEFAULT
        print("WARNING: Feldman algorithm does not yet handle multiple sensitive attrs.")

    def run(self, train_df, test_df, class_attr, sensitive_attrs, params):
        repaired_train_df = self.repair(train_df, sensitive_attrs)

        # What should be happening here is that the test_df is transformed using exactly the same
        # transformation as the train_df.  This will only be the case based on the usage below if
        # the distribution of each attribute conditioned on the sensitive attribute is the same
        # in the training set and the test set.
        repaired_test_df = self.repair(test_df, sensitive_attrs) 

        return self.model.run(repaired_train_df, repaired_test_df, class_attr, sensitive_attrs, 
                              params)
       
    def repair(self, data_df, sensitive_attrs):
        data = data_df.values.tolist()

        ## TODO: do something to make joint distribution over sensitive attrs
        index_to_repair = data_df.columns.get_loc(sensitive_attrs[0])
        headers = data_df.columns.tolist()
        repairer = Repairer(data, index_to_repair, self.repair_level, False)
       
        # The repaired data no longer includes its headers. 
        data = repairer.repair(data)
        data_df = DataFrame(data, columns = headers)
        return data_df
        
    def numerical_data_only(self):
        """
        The Feldman algorithm can preprocess both numerical and categorical data, the limiting
        factor is the capacity of the model that data is then passed to.
        """
        return self.model.numerical_data_only() 
