from algorithms.Algorithm import Algorithm
import numpy
import tempfile
import os
import subprocess

class KamishimaAlgorithm(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        self.name = "Kamishima"

    def run(self, train_df, test_df, class_attr, sensitive_attrs, single_sensitive, params):
        if len(sensitive_attrs) > 1:
            print("THIS CAN ONLY HANDLE ONE SENSITIVE ATTRIBUTE, PANIC")
            exit(10000)

        def create_file_in_kamishima_format(df):
            y = df[class_attr]
            s = df[sensitive_attrs]

            x = []
            for col in df:
                if col == class_attr:
                    continue
                if col in sensitive_attrs:
                    continue
                x.append(numpy.array(df[col].values, dtype=numpy.float64))

            s_dict = dict((k, i)
                            for (i, k) in enumerate(
                v[0] for v in s.drop_duplicates().values.tolist()))
            s_numeric_values = numpy.array(list(s_dict[s_value[0]] for s_value in s.values),
                                           dtype=numpy.float64)

            x.append(s_numeric_values)
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
        subprocess.run(['python3', './algorithms/kamishima/kamfadm-2012ecmlpkdd/train_pr.py',
                        '-i', train_name,
                        '-o', model_name])
        subprocess.run(['python3', './algorithms/kamishima/kamfadm-2012ecmlpkdd/predict_lr.py',
                        '-i', test_name,
                        '-m', model_name,
                        '-o', output_name])
        os.unlink(train_name)
        os.unlink(model_name)
        os.unlink(test_name)

        m = numpy.loadtxt(output_name)
        os.unlink(output_name)

        return m[:,1]

    def numerical_data_only(self):
        return True

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
