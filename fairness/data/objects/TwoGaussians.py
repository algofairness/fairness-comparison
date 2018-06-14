from fairness.data.objects.Data import Data
import numpy as np
import pandas as pd

##############################################################################

class TwoGaussians(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'two-gaussians'
        self.class_attr = 'decision'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sensitive-attr']
        self.privileged_class_names = ['privileged']
        self.categorical_features = []
        self.features_to_keep = ['a1', 'sensitive-attr', 'decision']
        self.missing_val_indicators = ['?']

    def load_raw_dataset(self):
        a1_g1 = np.random.randn(100)
        a1_g2 = np.random.randn(100) + 0.5
        sensitive_attr_g1 = np.full(a1_g1.shape, 'non-privileged')
        sensitive_attr_g2 = np.full(a1_g2.shape, 'privileged')
        a1 = np.concatenate((a1_g1, a1_g2))
        sensitive_attr = np.concatenate((sensitive_attr_g1, sensitive_attr_g2))
        # There has to be a better way of dong this.
        decision = [1 if v > 0 else 0 for v in a1 > 0]
        return pd.DataFrame(data={
            "decision": decision,
            "sensitive-attr": sensitive_attr,
            "a1": a1
            })

        
