from abc import ABCMeta, abstractmethod

class AbstractRepairer(object):
  """
  A Repairer object is capable of removing the correlations of a feature
  from a dataset at a specific `repair_level`.
  """

  __metaclass__ = ABCMeta

  def __init__(self, all_data, feature_to_repair, repair_level, features_to_ignore=[]):
    """
    all_data should be a list of rows (ie, a list of lists) composing the entire
    test and training dataset. Headers should not be included in data sets.

    feature_to_repair should be the index of the feature to repair. (ie, 0 to k)
    where k is the number of features in the dataset.

    repair_level should be a float between [0,1] representing the level of repair.

    features_to_ignore should be a list of feature indexes that should be ignored.
    """

    self.all_data = all_data
    self.feature_to_repair = feature_to_repair
    self.repair_level = repair_level
    self.features_to_ignore = features_to_ignore

  @abstractmethod
  def repair(self, data_to_repair):
    """
    data_to_repair is the list of rows that actually should be repaired.
    """
    pass
