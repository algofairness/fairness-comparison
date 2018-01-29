from algorithms.Algorithm import Algorithm
import numpy
import tempfile
import os
import subprocess

class CaldersAlgorithm(Algorithm):
    """
    Notes:

    - The original code depends on python2's commands library. We hacked
    it to hve python3 support by adding a minimal commands.py module with
    a getoutput function.

    ## Getting train_pr to work

    It takes as input a space-separated file.

    Its value imputation is quite naive (replacing nans with column
    means), so we will impute values ourselves ahead of time if necessary.

    The documentation describes 'ns' as the number of sensitive features,
    but the code hardcodes ns=1, and things only seem to make sense if
    'ns' is, instead, the column _index_ for the sensitive feature,
    _counting from the end, and excluding the target class_. In addition,
    it seems that if the sensitive feature is not the last column of the
    data, the code will drop all features after that column.

    tl;dr:

    - the last column of the input should be the target class (as integer values),
    - the code only appears to support one sensitive feature at a time,
    - the second-to-last column of the input should be the sensitive feature (as integer values)
    - fill missing values ahead of time in order to avoid imputation.

    If you do this, train_pr.py:148-149 will take the last column to be y
    (the target classes to predict), then pr.py:264 will take the
    second-to-last column as the sensitive attribute, and pr.py:265-268
    will take the remaining columns as non-sensitive.

    """

    def __init__(self):
        Algorithm.__init__(self)
        self.name = "Calders"

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):

        if not 'beta' in params:
            params = self.get_default_params()

        class_type = type(train_df[class_attr].values[0].item())

        def create_file_in_kamishima_format(df):
            y = df[class_attr]
            s = df[single_sensitive]

            x = []
            for col in df:
                if col == class_attr:
                    continue
                if col in sensitive_attrs:
                    continue
                x.append(numpy.array(df[col].values, dtype=numpy.float64))

            x.append(numpy.array(s, dtype=numpy.float64))
            x.append(numpy.array(df[class_attr], dtype=numpy.float64))

            result = numpy.array(x).T
            fd, name = tempfile.mkstemp()
            os.close(fd)
            numpy.savetxt(name, result)
            return name

        fd, model_name = tempfile.mkstemp()
        os.close(fd)
        fd, output_name = tempfile.mkstemp()
        os.close(fd)
        train_name = create_file_in_kamishima_format(train_df)
        test_name = create_file_in_kamishima_format(test_df)
        beta_val = params['beta']
        subprocess.run(['python3', './algorithms/kamishima/kamfadm-2012ecmlpkdd/train_cv2nb.py',
                        '-b', str(beta_val),
                        '-i', train_name,
                        '-o', model_name,
                        '--quiet'])
        subprocess.run(['python3', './algorithms/kamishima/kamfadm-2012ecmlpkdd/predict_nb.py',
                        '-i', test_name,
                        '-m', model_name,
                        '-o', output_name,
                        '--quiet'])
        os.unlink(train_name)
        os.unlink(model_name)
        os.unlink(test_name)

        m = numpy.loadtxt(output_name)
        os.unlink(output_name)

        predictions = m[:,1]
        predictions_correct = [class_type(x) for x in predictions]

        return predictions_correct

    def numerical_data_only(self):
        return True

    def binary_sensitive_attrs_only(self):
        return True

    def get_default_params(self):
        """
        According to the source code, 
        'coefficient of a prior for feature probabilities (default 1.0)'
        """
        return { 'beta' : 1.0 }

    def get_param_info(self):
        """
        FIXME this is probably completely wrong.
        """
        return {'beta' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}


