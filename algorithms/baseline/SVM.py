from sklearn.svm import SVC as SKLearn_SVM
from algorithms.baseline.Generic import Generic

class SVM(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_SVM()
        self.name = "SVM"
