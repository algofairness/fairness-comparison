from fairness.metrics.Metric import Metric
from sklearn.metrics import confusion_matrix

class TNR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'TNR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        classes = list(set(actual))
        matrix = confusion_matrix(actual, predicted, labels=classes)
        # matrix[i][j] is the number of observations with actual class i that were predicted as j
        TN = 0.0
        allN = 0.0
        for i in range(0, len(classes)):
            trueval = classes[i]
            if trueval == positive_pred:
                continue
            for j in range(0, len(classes)):
                allN += matrix[i][j]
                predval = classes[j]
                if trueval == predval:
                    TN += matrix[i][j]

        if allN == 0.0:
            return 1.0

        return TN / allN
