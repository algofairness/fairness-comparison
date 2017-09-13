from abc import ABCMeta, abstractmethod
import os, sys
from preprocessing.prepare_german import prepare_german
from preprocessing.prepare_adult import prepare_adult
from preprocessing.prepare_compas import prepare_compas
import numpy as np
import pandas as pd
from misc.black_box_auditing import *
from data.propublica.load_numerical_compas import *
from data.german.load_german_data import *
from data.adult.load_adult import *
from metrics.metrics import *

class AbstractAlgorithm(object, metaclass=ABCMeta):
  def __init__(self, data, params, prepare, classify):
    """ data: str for which data set to use
	params: dict with param name as key and val as val (ex: {"eta": 30})
    """
    self.data = data
    self.params = params
    self.prepare = prepare
    self.classify = classify
    
    """
    if data == "adult":
      self.prepare = prepare_adult
      self.name = "sex_adult"
      self.filename = "feldmen_cleaned_sex_adult_nb_0"
      self.classify = classify_adult
    if data == "compas":
      self.prepare = prepare_compas
      self.name = "propublica"
      self.filename = "propublica_race_nb_0"
      self.classify = classify_compas
    if data == "german":
      self.prepare = prepare_german
      self.name = "german"
      self.filename = "german_sex_nb_0"
      self.classify = classify_german
    """

    self.x_train, self.y_train, self.x_control_train, self.x_test, self.y_test, self.x_control_test, self.sensitive_attr = self.prepare()

  @abstractmethod
  def run(self):
    """ runs algorithm and returns inputs for Metrics """
    pass
  
