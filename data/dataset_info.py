# Datasets by keyword - these will be used throughout as dictionary indicies and are also the
# assumed stem for file names (e.g., 'german.csv').
DATASETS = ['german', 'adult', 'ricci', 'retailer']

# The name of any sensitive / protected attribute(s) that will be used for a fairness analysis
# and should not be used to train the model.
SENSITIVE_ATTRS = { 'german' : ['sex'] ,
                    'adult' : ['race', 'sex'] ,
                    'ricci' : ['Race'] ,
                    'retailer' : ['urace_orig']
                  }

UNPROTECTED_CLASS_NAMES = { 'german' : ['M'] , # TODO: check this
                           'adult' : ['White', 'Male'] ,
                           'ricci' : ['W'] ,
                           'retailer' : ['White']  # TODO: check this
                          }

# Feature types per data set - numerical (float), ordered categorical (to be converted to int),
# and categorical (string).
# TODO: gather this information for german and retailer
FEATURE_TYPES = { 'ricci' : [ 'cat', 'num', 'num', 'cat', 'num' ] ,
                  'adult' : [ 'num', 'cat', 'num', 'ord', 'num', 'cat', 'cat', 'cat', 'cat', 'cat',
                              'num', 'num', 'num', 'cat', 'cat' ]
                }

# TODO: determine the actual feature names for the retailer data set.
FEATURES_TO_KEEP = { 'german' : [] ,
                     'adult' : [] ,
                     'ricci' : [] ,
                     'retailer' : [ 'applicant_id' ]
                   }  

RAW_DATA_DIR = '../data/raw/'
PROCESSED_DATA_DIR = '../data/preprocessed/'
