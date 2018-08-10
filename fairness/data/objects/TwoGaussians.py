from fairness.data.objects.Data import Data
import numpy as np
import pandas as pd
import math

##############################################################################

TOTAL_ITEMS = 200

class TwoGaussians(Data):

    def __init__(self, percent_pos):
        Data.__init__(self)
        self.dataset_name = 'two-gaussians_' + str(percent_pos)
        self.class_attr = 'decision'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sensitive-attr']
        self.privileged_class_names = ['privileged']
        self.categorical_features = []
        self.features_to_keep = ['a1', 'sensitive-attr', 'decision']
        self.missing_val_indicators = ['?']
        self.num_pos_class = math.floor(percent_pos * TOTAL_ITEMS)

    def load_raw_dataset(self):
        a1_g1 = np.random.randn(math.floor(TOTAL_ITEMS / 2))
        a1_g2 = np.random.randn(math.floor(TOTAL_ITEMS / 2)) + 0.5
        sensitive_attr_g1 = np.full(a1_g1.shape, 'non-privileged')
        sensitive_attr_g2 = np.full(a1_g2.shape, 'privileged')
        a1 = np.concatenate((a1_g1, a1_g2))
        sensitive_attr = np.concatenate((sensitive_attr_g1, sensitive_attr_g2))
        a_sorted = sorted(a1)
        threshold = a_sorted[TOTAL_ITEMS - 1 - self.num_pos_class]
        # There has to be a better way of dong this.
        decision = [1 if v > threshold else 0 for v in a1]
        positive_count = sum(1 for x in decision if x == 1)
        negative_count = sum(1 for x in decision if x == 0)
        return pd.DataFrame(data={
            "decision": decision,
            "sensitive-attr": sensitive_attr,
            "a1": a1
            })


