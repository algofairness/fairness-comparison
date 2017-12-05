# Datasets by keyword - these will be used throughout as dictionary indicies and are also the
# assumed stem for file names (e.g., 'german.csv').
DATASETS = ['german', 'adult', 'ricci']  ## TODO: add in , 'retailer']

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

# Features that should be expanded to one-hot versions for numerical-only algorithms.  This
# should not include the protected features or the outcome class variable.
# TODO: include german and retailer
CATEGORICAL_FEATURES = { 'ricci' : [ 'Position' ] ,
                         'adult' : [ 'workclass', 'marital-status', 'occupation', 'relationship',
                                     'native-country' ] ,
                         'german' : [] ,
                         'retailer' : []
                       }

FEATURES_TO_KEEP = { 'german' : [ 'status', 'month', 'credit_history', 'purpose', 'credit_amount',
                                  'savings', 'employment', 'investment_as_income_percentage', 
                                  'personal_status', 'other_debtors', 'residence_since',
                                  'property', 'age', 'installment_plans', 'housing', 
                                  'number_of_credits', 'skill_level', 'people_liable_for',
                                  'telephone', 'foreign_worker', 'credit' ],
                     'adult' : [ 'age', 'workclass', 'education', 'education-num', 'marital-status',
                                 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                 'capital-loss', 'hours-per-week', 'native-country',
                                 'income-per-year' ] ,
                     'ricci' : [ 'Position', 'Oral', 'Written', 'Race', 'Combine' ] ,
                     'retailer' : [ 'usite', 'azip', 'urace_orig', 'udateofbirth',
                                    'ugender', 'szip', 'csvr2', 'hired' ]
                   }  
