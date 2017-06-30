from abc import ABCMeta, abstractmethod
import time

class AbstractModelFactory(object):
  __metaclass__ = ABCMeta

  def __init__(self, all_data, headers, response_header, name_prefix="",
               features_to_ignore=[], options={}):
    self.all_data = all_data
    self.headers = headers
    self.response_header = response_header
    self.features_to_ignore=features_to_ignore
    self.factory_name = "{}_{}".format(name_prefix, time.time()) if name_prefix else time.time()

    # All `options` should be consumed by the time the Abstract Factory is called.
    if options:
      raise Exception("Unknown ModelFactory options set: {}".format(options))

  @abstractmethod
  def build(self, train_set):
    pass
