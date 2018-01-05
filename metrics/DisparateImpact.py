from FairnessMetric import FairnessMetric

class DisparateImpact(FairnessMetric):
    """
    This metric calculates disparate imapct in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    """
    def __init__(self, actual, predicted, sensitive, unprotected_vals):
        FairnessMetric.__init__(actual, predicted, sensitive, unprotected_vals, positive_pred)
        self.name = 'disparate impact'

    def calc(self):
        # This implementation assumes that predicted and sensitive both have the same lengths.
        unprotected_positive = 0.0
        unprotected_negative = 0.0
        protected_positive = 0.0
        protected_negative = 0.0
        for i in range(0, len(self.predicted)):
            protected_val = self.sensitive[i]
            predicted_val = self.predicted[i]
            if protected_val in self.unprotected_vals:
                if predicted_val == self.positive_pred:
                    unprotected_positive += 1
                else:
                    unprotected_negative += 1
            else:
                if predicted_val == self.positive_pred:
                    protected_positive += 1
                else:
                    protected_negative += 1
  
        protected_pos_percent = protected_positive / (protected_positive + protected_negative)
        unprotected_pos_percent = unprotected_positive / 
                                  (unprotected_positive + unprotected_negative)
        return protected_pos_percent / unprotected_pos_percent
            
                
        
