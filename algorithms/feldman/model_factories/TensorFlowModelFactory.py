from AbstractModelFactory import AbstractModelFactory
from AbstractModelVisitor import AbstractModelVisitor

import os
import numpy as np
import tensorflow as tf
import time

TMP_DIR = "tmp/"
if not os.path.exists(TMP_DIR):
  os.makedirs(TMP_DIR)

class ModelFactory(AbstractModelFactory):

  def __init__(self, *args, **kwargs):
    self.num_epochs = 100
    self.batch_size = 100
    self.learning_rate = 0.01
    self.hidden_layer_sizes = [] # If empty, no hidden layers are used.
    self.layer_types = [tf.nn.softmax] # The first layer is the input layer.

    if "options" in kwargs:
      options = kwargs["options"]
      if "num_epochs"  in options:
        self.num_epochs = options.pop("num_epochs")
      if "batch_size"  in options:
        self.batch_size = options.pop("batch_size")
      if "learning_rate"  in options:
        self.learning_rate = options.pop("learning_rate")
      if "hidden_layer_sizes"  in options:
        self.hidden_layer_sizes = options.pop("hidden_layer_sizes")
      if "layer_types"  in options:
        # Import the appropriate layer.
        layer_names = options.pop("layer_types")
        self.layer_types = [getattr(tf.nn, layer) for layer in layer_names]

    super(ModelFactory, self).__init__(*args, **kwargs)

    self.verbose_factory_name = "TensorFlow_Network"
    self.response_index = self.headers.index(self.response_header)

    #TODO: Make this nicer, ye unholy PEP-traitor.
    val_set = {h:{r[i] for r in self.all_data} for i,h in enumerate(self.headers)}

    # Mark any categorical features for column expansion.
    headers_to_translate = []
    for i, header in enumerate(self.headers):
      categorical = all(type(val)==str for val in val_set[header])
      if i == self.response_index or categorical:
        headers_to_translate.append(header)

    #TODO: Make this nicer, ye unholy PEP-traitor.
    self.trans_dict = {h:{v:i for i,v in enumerate(vals)} for h, vals in val_set.items() if h in headers_to_translate}

    #TODO: Make this nicer.
    self.normalizers = {h:{
      "mean": len(val_set[h]) if all(type(e)==str for e in val_set[h]) else sum(r[i] for r in self.all_data)/float(len(self.all_data)),
      "max": None if all(type(e)==str for e in val_set[h]) else max(val_set[h]),
      "min": None if all(type(e)==str for e in val_set[h]) else min(val_set[h]),
      } for i,h in enumerate(self.headers)}

    # If the response column is shifted by expanding categorical features,
    # update where the response index should be.
    response_col_shift = 0
    for header in self.headers[:self.response_index]:
      if header in self.trans_dict:
        response_col_shift += len(self.trans_dict[header])-1
    self.relative_response_index = self.response_index + response_col_shift

    self.num_labels = len(val_set[self.response_header])

  def build(self, train_set): #TODO: Add a features-to-ignore option.
    # In case the class is a string, translate it.
    translated_train_set = translate_dataset(self.response_index, train_set, self.trans_dict, self.headers, self.normalizers)

    train_matrix, train_labels = list_to_tf_input(translated_train_set, self.relative_response_index, self.num_labels)
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
    train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)

    saver = tf.train.Saver()  # Defaults to saving all variables.

    # Create a local session to run this computation.
    with tf.Session() as tf_session:
      # For the test data, hold the entire dataset in one constant node.
      tf.global_variables_initializer().run()

      # Iterate and train.
      for step in xrange(self.num_epochs * train_size // self.batch_size):

        offset = (step * self.batch_size) % train_size
        batch_data = train_matrix[offset:(offset + self.batch_size), :]
        batch_labels = train_labels[offset:(offset + self.batch_size)]
        train_step.run(feed_dict={x: batch_data, y_: batch_labels})

        # Save the model file each step.
        model_name="{}/{}_{}_{}.model".format(TMP_DIR, self.verbose_factory_name, self.factory_name, time.time())
        checkpoint = saver.save(tf_session, model_name, global_step=step+1)

    return ModelVisitor(model_name, checkpoint, saver, self.response_header, self.response_index, self.relative_response_index, self.num_labels, x, y_, y, self.trans_dict, self.headers, self.normalizers)


class ModelVisitor(AbstractModelVisitor):

  def __init__(self, model_name, checkpoint, model_saver, response_header, response_index, relative_response_index, num_labels, x, y_, y, trans_dict, headers, normalizers):
    super(ModelVisitor,self).__init__(model_name)
    self.model_saver = model_saver
    self.checkpoint = checkpoint
    self.response_index = response_index
    self.response_header = response_header
    self.relative_response_index = relative_response_index
    self.num_labels = num_labels
    self.x = x
    self.y_ = y_
    self.y = y
    self.trans_dict = trans_dict
    self.headers = headers
    self.normalizers = normalizers

  def test(self, test_set, test_name=""):
    translated_test_set = translate_dataset(self.response_index, test_set, self.trans_dict, self.headers, self.normalizers)

    test_matrix, test_labels = list_to_tf_input(translated_test_set, self.relative_response_index, self.num_labels  )

    with tf.Session() as tf_session:
      self.model_saver.restore(tf_session, self.checkpoint)
      predictions = tf.argmax(self.y, 1).eval(feed_dict={self.x: test_matrix, self.y_:test_labels}, session=tf_session)

    predictions_dict = {i:key for key,i in self.trans_dict[self.response_header].items()}
    predictions = [predictions_dict[pred] for pred in predictions]

    return zip([row[self.response_index] for row in test_set], predictions)

def list_to_tf_input(data, response_index, num_labels):
  matrix = np.matrix([row[:response_index] + row[response_index+1:] for row in data])

  labels = np.asarray([row[response_index] for row in data], dtype=np.uint8)
  labels_onehot = (np.arange(num_labels) == labels[:, None]).astype(np.float32)

  return matrix, labels_onehot

def translate_dataset(response_index, data_set, trans_dict, headers, normalizers):
  translated_set = []
  for i, row in enumerate(data_set):
    new_row = []
    for j, val in enumerate(row):
      header = headers[j]
      if j==response_index:
        translated = trans_dict[header][val]
        new_row.append(translated)
      elif header in trans_dict:
        translated = trans_dict[header][val]
        val_list = [0]* len(trans_dict[header])
        val_list[translated] = 1
        new_row.extend(val_list)
      else:
        norm = normalizers[header]
        if (norm["max"]-norm["min"]) > 0:
          normed = (val-norm["mean"])/(norm["max"]-norm["min"])
        else:
          normed = 0.0
        new_row.append(normed)
    translated_set.append(new_row)
  return translated_set

def test():
  test_categorical_model()
  test_categorical_response()
  test_list_to_tf_input()
  test_basic_model()

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

def test_categorical_response():
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

def test_categorical_model():
  headers = ["predictor", "response"]
  response = "response"
  train_set = [["A","A"] for i in range(1,50)] + [["B","B"] for i in range(1,50)]
  test_set = [["A","A"] for i in range(1,50)] + [["B","C"] for i in range(1,50)]
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
