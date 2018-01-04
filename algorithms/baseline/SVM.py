from sklearn.svm import SVC as SKLearn_SVM
from algorithms.Algorithm import Algorithm

class SVM(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)

    def run(self, train_df, test_df, class_attr, sensitive_attrs, params):
        # remove sensitive attributes from the training set
        train_df_nosensitive = train_df.drop(columns = sensitive_attrs)
        test_df_nosensitive = test_df.drop(columns = sensitive_attrs)

        # create and train the classifier
        classifier = SKLearn_SVM()
        y = train_df_nosensitive[class_attr]
        X = train_df_nosensitive.drop(columns = class_attr)
        classifier.fit(X, y)

        # get the predictions on the test set
        X_test = test_df_nosensitive.drop(class_attr, axis=1)
        predictions = classifier.predict(X_test)

        # get the actual classifications and sensitive attributes
        actual = test_df[class_attr]
        sensitive = test_df[sensitive_attrs]

        return actual, predictions, sensitive

    def numerical_data_only(self):
        """
        Returns True if this algorithm can only handle numerical data as input.
        """
        return True
