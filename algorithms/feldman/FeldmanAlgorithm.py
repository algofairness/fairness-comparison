import BlackBoxAuditing as BBA
from algorithms.Algorithm import Algorithm

REPAIR_LEVEL_DEFAULT = 1.0

class FeldmanAlgorithm(Algorithm):
    def __init__(self, algorithm, repair_level=REPAIR_LEVEL_DEFAULT):
        Algorithm.__init__(self)
        self.model = algorithm
        self.repair_level = REPAIR_LEVEL_DEFAULT

    def run(self, train_df, test_df, class_attr, sensitive_attrs, params):
            
        train_data = train_df.values.tolist()
        ## TODO: do something to make joint distribution over sensitive attrs
        index_to_repair = train_df.columns.get_loc(sensitive_attr)
        repairer = Repairer(train_data, index_to_repair, self.repair_level, False)
        train_data = repairer.repair(train_data)
        train_df = DataFrame(train_data)

        ## TODO: are we supposed to do something here with the training data too?
        print("WARNING: Feldman algorithm may not be fully implemented.")

        return model.run(train_df, test_df, class_attr, sensitive_attrs, params)
        
        
