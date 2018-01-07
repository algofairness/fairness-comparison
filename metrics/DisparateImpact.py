from metrics.FairnessMetric import FairnessMetric

class DisparateImpact(FairnessMetric):
    """
    This metric calculates disparate imapct in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    If there are no positive protected classifications, 0.0 is returned. 
    """
    def __init__(self):
        FairnessMetric.__init__(self)
        self.name = 'disparate impact'

    def calc(self, actual, predicted, sensitive, unprotected_vals, positive_pred):
        # This implementation assumes that predicted and sensitive both have the same lengths.
        unprotected_positive = 0.0
        unprotected_negative = 0.0
        protected_positive = 0.0
        protected_negative = 0.0
        for i in range(0, len(predicted)):
            protected_val = sensitive[i]
            predicted_val = predicted[i]
            if protected_val in unprotected_vals:
                if str(predicted_val) == str(positive_pred):
                    unprotected_positive += 1
                else:
                    unprotected_negative += 1
            else:
                if str(predicted_val) == str(positive_pred):
                    protected_positive += 1
                else:
                    protected_negative += 1

        protected_pos_percent = 0.0
        if protected_positive + protected_negative > 0: 
            protected_pos_percent = protected_positive / (protected_positive + protected_negative)
        unprotected_pos_percent = 0.0
        if unprotected_positive + unprotected_negative > 0:
            unprotected_pos_percent = unprotected_positive /  \
                                      (unprotected_positive + unprotected_negative)
        DI = 0.0
        if unprotected_pos_percent > 0:
            DI = protected_pos_percent / unprotected_pos_percent
        return DI
