from AbstractModelFactory import AbstractModelFactory
from AbstractModelVisitor import AbstractModelVisitor

import os
import numpy as np
import tensorflow as tf
import datetime

TMP_DIR = "tmp/"
CHECKPOINT_DIR = "tmp/tensorflow_checkpoints/"
for directory in [TMP_DIR, CHECKPOINT_DIR]:
  if not os.path.exists(directory):
    os.makedirs(directory)


class ModelFactory(AbstractModelFactory):

  def __init__(self, *args, **kwargs):
    super(ModelFactory, self).__init__(*args, **kwargs)
    self.verbose_factory_name = "TensorFlow_Network"
    self.model_name = "Recidivism TensorFlow"

    self.num_epochs = 2000
    self.batch_size = 500

    self.response_index = self.headers.index(self.response_header)

    possible_values = set(row[self.response_index] for row in self.all_data)
    self.num_labels = len(possible_values)
    self.response_dict = {val:i for i,val in enumerate(possible_values)}

    self.hidden_layer_sizes = [100, 40] # If empty, no hidden layers are used.
    self.layer_types = [tf.nn.softmax,
			tf.nn.softmax,   # Input Layer
                        tf.nn.softmax]     # 4th Hidden Layer

  def build(self, train_set): #TODO: Add a features-to-ignore option.
    # In case the class is a string, translate it.
    translated_train_set = translate_response(self.response_index, train_set, self.response_dict)

    train_matrix, train_labels = list_to_tf_input(translated_train_set, self.response_index, self.num_labels)
    train_size, num_features = train_matrix.shape

    # Construct the layer architecture.
    x = tf.placeholder("float", shape=[None, num_features]) # Input
    y_ = tf.placeholder("float", shape=[None, self.num_labels]) # Output.

    layer_sizes = [num_features] + self.hidden_layer_sizes + [self.num_labels]
    # Generate a layer for the input and for each additional hidden layer.
    layers = [x] # Count the input as the first layer.
    for i in xrange(len(layer_sizes)-1):
      layer_size = layer_sizes[i]
      layer_type = self.layer_types[i]

      prev_layer = layers[-1]
      next_layer_size = layer_sizes[i+1]

      # Create a new layer with initially random weights and biases.
      W = tf.Variable(tf.random_normal([layer_size, next_layer_size]))
      b = tf.Variable(tf.random_normal([next_layer_size]))
      new_layer = layer_type(tf.matmul(prev_layer, W) + b)

      layers.append( new_layer )

    y = layers[-1]

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    saver = tf.train.Saver()  # Defaults to saving all variables.

    # Create a local session to run this computation.
    with tf.Session() as tf_session:
      # For the test data, hold the entire dataset in one constant node.
      tf.initialize_all_variables().run()

      # Iterate and train.
      for step in xrange(self.num_epochs * train_size // self.batch_size):

        offset = (step * self.batch_size) % train_size
        batch_data = train_matrix[offset:(offset + self.batch_size), :]
        batch_labels = train_labels[offset:(offset + self.batch_size)]
        train_step.run(feed_dict={x: batch_data, y_: batch_labels})

        # Save the model file each step.
        saver.save(tf_session, CHECKPOINT_DIR + 'model.ckpt', global_step=step+1)

    return ModelVisitor(saver, self.response_index, self.num_labels, x, y_, y, self.response_dict)


class ModelVisitor(AbstractModelVisitor):

  def __init__(self, model_saver, response_index, num_labels, x, y_, y, response_dict):
    self.model_saver = model_saver
    self.model_name = "Recidivism TensorFlow Visitor"
    self.response_index = response_index
    self.num_labels = num_labels
    self.x = x
    self.y_ = y_
    self.y = y
    self.response_dict = response_dict

  def test(self, test_set, test_name=""):
    translated_test_set = translate_response(self.response_index, test_set, self.response_dict)

    test_matrix, test_labels = list_to_tf_input(translated_test_set, self.response_index, self.num_labels  )

    with tf.Session() as tf_session:
      ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
      self.model_saver.restore(tf_session, ckpt.model_checkpoint_path)
      predictions = tf.argmax(self.y, 1).eval(feed_dict={self.x: test_matrix, self.y_:test_labels}, session=tf_session)

    predictions_dict = {i:key for key,i in self.response_dict.items()}
    predictions = [predictions_dict[pred] for pred in predictions]

    return zip([row[self.response_index] for row in test_set], predictions)

def list_to_tf_input(data, response_index, num_labels):
  data = expand_to_one_hot(data)
  matrix = np.matrix([row[:response_index] + row[response_index+1:] for row in data])

  labels = np.asarray([row[response_index] for row in data], dtype=np.uint8)
  labels_onehot = (np.arange(num_labels) == labels[:, None]).astype(np.float32)

  return matrix, labels_onehot

def translate_response(response_index, data_set, response_dict):
  translated_set = []
  for row in data_set:
    response = row[response_index]
    translation = response_dict[response]
    new_row = row[:response_index]+[translation]+row[response_index+1:]
    translated_set.append(new_row)
  return translated_set



def expand_to_one_hot(data,expand = True,use_alternative=False):
    header_dict = {'ALCABUS':0,'PRIRCAT':1,'TMSRVC':2,'SEX1':3,'RACE':4,'RELTYP':5,'age_1st_arrest':6,'DRUGAB':7,'Class':8,'RLAGE':9,'NFRCTNS':10}

    new_data = []
    for entry in data:
	temp = {}
	if expand == True:
	    if entry[header_dict["SEX1"]] == "FEMALE":
		temp['female'] = 1
	    else:
		temp['female'] = 0

	    if entry[header_dict["ALCABUS"]] == 'INMATE IS AN ALCOHOL ABUSER':
		temp['prior_alcohol_abuse'] = 1
	    else:
		temp['prior_alcohol_abuse'] = 0

	    if entry[header_dict['DRUGAB']] == 'INMATE IS A DRUG ABUSER':
		temp['prior_drug_abuse'] = 1
	    else:
		temp['prior_drug_abuse'] = 0

	    if entry[header_dict['NFRCTNS']] == 'INMATE HAS RECORD':
		temp['infraction_in_prison'] = 1
	    else:
		temp['infraction_in_prison'] = 0

	    race_cats = ['WHITE','BLACK','AMERICAN INDIAN/ALEUTIAN','ASIAN/PACIFIC ISLANDER','OTHER','UNKNOWN']

	    for cat in race_cats:
		if entry[header_dict['RACE']] == cat:
		    temp['race_'+cat] = 1
		else:
		    temp['race_'+cat] = 0

	    release_age_cats = ['14 TO 17 YEARS OLD','18 TO 24 YEARS OLD', '25 TO 29 YEARS OLD', \
	    '30 TO 34 YEARS OLD','35 TO 39 YEARS OLD','40 TO 44 YEARS OLD','45 YEARS OLD AND OLDER']
	    for cat in release_age_cats:
		if entry[header_dict['RLAGE']] == cat:
		    temp['release_age_'+cat] = 1
		else:
		    temp['release_age_'+cat] = 0

	    time_served_cats = ['None','1 TO 6 MONTHS','13 TO 18 MONTHS','19 TO 24 MONTHS','25 TO 30 MONTHS', \
			'31 TO 36 MONTHS','37 TO 60 MONTHS','61 MONTHS AND HIGHER','7 TO 12 MONTHS']
	    for cat in time_served_cats:
		if entry[header_dict['TMSRVC']] == cat:
		    temp['time_served_'+cat] = 1
		else:
		    temp['time_served_'+cat] = 0

	    prior_arrest_cats = ['None','1 PRIOR ARREST','11 TO 15 PRIOR ARRESTS','16 TO HI PRIOR ARRESTS','2 PRIOR ARRESTS', \
		'3 PRIOR ARRESTS','4 PRIOR ARRESTS','5 PRIOR ARRESTS','6 PRIOR ARRESTS','7 TO 10 PRIOR ARRESTS']
	    for cat in prior_arrest_cats:
		if entry[header_dict['PRIRCAT']] == cat:
		    temp['prior_arrest_'+cat] = 1
		else:
		    temp['prior_arrest_'+cat] = 0

	    conditional_release =['PAROLE BOARD DECISION-SERVED NO MINIMUM','MANDATORY PAROLE RELEASE', 'PROBATION RELEASE-SHOCK PROBATION', \
			'OTHER CONDITIONAL RELEASE']
	    unconditional_release = ['EXPIRATION OF SENTENCE','COMMUTATION-PARDON','RELEASE TO CUSTODY, DETAINER, OR WARRANT', \
			'OTHER UNCONDITIONAL RELEASE']
	    other_release = ['NATURAL CAUSES','SUICIDE','HOMICIDE BY ANOTHER INMATE','OTHER HOMICIDE','EXECUTION','OTHER TYPE OF DEATH', \
		    'TRANSFER','RELEASE ON APPEAL OR BOND','OTHER TYPE OF RELEASE','ESCAPE','ACCIDENTAL INJURY TO SELF','UNKNOWN']
	    if entry[header_dict['RELTYP']] in conditional_release:
		temp['released_conditional'] = 1
		temp['released_unconditional'] = 0
		temp['released_other'] = 0
	    elif entry[header_dict['RELTYP']] in unconditional_release:
		temp['released_conditional'] = 0
		temp['released_unconditional'] = 1
		temp['released_other'] = 0
	    else:
		temp['released_conditional'] = 0
		temp['released_unconditional'] = 0
		temp['released_other'] = 1

	    first_arrest_cats = ['UNDER 17','BETWEEN 18 AND 24','BETWEEN 25 AND 29','BETWEEN 30 AND 39','OVER 40']
	    for cat in first_arrest_cats:
		if entry[header_dict['age_1st_arrest']] == cat:
		    temp['age_first_arrest_'+cat] = 1
		else:
		    temp['age_first_arrest_'+cat] = 0
	else:
	    temp['SEX1'] = entry['SEX1']
	    temp['RELTYP'] = entry['RELTYP']
	    temp['PRIRCAT'] = entry['PRIRCAT']
	    temp['ALCABUS'] = entry['ALCABUS']
	    temp['DRUGAB'] = entry['DRUGAB']
	    temp['RLAGE'] = entry['RLAGE']
	    temp['TMSRVC'] = entry['TMSRVC']
	    temp['NFRCTNS'] = entry['NFRCTNS']
	    temp['RACE'] = entry['RACE']
	    try:
		bdate = datetime.date(int(entry['YEAROB2']),int(entry['MNTHOB2']), int(entry['DAYOB2']))
		first_arrest = datetime.date(int(entry['A001YR']),int(entry['A001MO']),int(entry['A001DA']))
		first_arrest_age = first_arrest - bdate
		temp['age_1st_arrest'] = first_arrest_age.days
	    except:
		temp['age_1st_arrest'] = 0

	new_data.append(temp)


    # convert from dictionary to list of lists
    fin = [[int(entry[key]) for key in entry.keys()] for entry in new_data]
    """
    with open("brandon_testing/test_"+str(time.clock())+".csv","w") as f:
	writer = csv.writer(f,delimiter=",")
	for row in fin:
	    writer.writerow(row)
    """

    return fin





def test():
  # expand_to_one_hot assumes the data is the recidivism data, so these tests can't be run
  # test_list_to_tf_input()
  # test_basic_model()
  # test_categorical_model()
  pass

def test_list_to_tf_input():
  data = [[0,0],[0,1],[0,2]]
  tf_matrix, tf_onehot = list_to_tf_input(data, 1, 3)
  correct_matrix = [[0],[0],[0]]
  correct_onehot = [[1,0,0], [0,1,0], [0,0,1]]
  print "list_to_tf_input matrix correct? --",np.array_equal(tf_matrix, correct_matrix)
  print "list_to_tf_input onehot correct? --",np.array_equal(tf_onehot, correct_onehot)

def test_basic_model():
  headers = ["predictor 1", "predictor 2", "response"]
  response = "response"
  train_set = [[i,0,0] for i in range(1,50)] + [[0,i,1] for i in range(1,50)]
  test_set = [[i,0,0] for i in range(1,50)] + [[0,i,1] for i in range(1,50)]
  all_data = train_set + test_set

  factory = ModelFactory(all_data, headers, response, name_prefix="test")
  print "factory settings valid? -- ",len(factory.hidden_layer_sizes)+1 == len(factory.layer_types)

  model = factory.build(train_set)
  print "factory builds ModelVisitor? -- ", isinstance(model, ModelVisitor)

  predictions = model.test(test_set)
  resp_index = headers.index(response)
  intended_predictions = [(row[resp_index], row[resp_index]) for row in test_set]
  print "predicting numeric categories correctly? -- ", predictions == intended_predictions

def test_categorical_model():
  headers = ["predictor 1", "predictor 2", "response"]
  response = "response"
  train_set = [[i,0,"A"] for i in range(1,50)] + [[0,i,"B"] for i in range(1,50)]
  test_set = [[i,0,"A"] for i in range(1,50)] + [[0,i,"C"] for i in range(1,50)]
  all_data = train_set + test_set

  factory = ModelFactory(all_data, headers, response, name_prefix="test")
  print "factory settings valid? -- ",len(factory.hidden_layer_sizes)+1 == len(factory.layer_types)

  model = factory.build(train_set)
  print "factory builds ModelVisitor? -- ", isinstance(model, ModelVisitor)

  predictions = model.test(test_set)
  resp_index = headers.index(response)
  intended_predictions = [(test_row[resp_index], train_row[resp_index]) for train_row, test_row in zip(train_set,test_set)]
  print "predicting string-categories correctly? -- ", predictions == intended_predictions

if __name__=="__main__":
  test()
