#!/usr/bin/env python3

import random
from collections import OrderedDict
from fairness.algorithms.Ben import boosting
from fairness.algorithms.Ben import svm
from fairness.algorithms.Ben import lr

from fairness.data.objects import Adult, German
from fairness.algorithms.Ben.weaklearners.decisionstump import buildDecisionStump
from fairness.algorithms.Ben import errorfunctions
from fairness.algorithms.Ben.utils import arrayErrorBars, errorBars, experimentCrossValidate
from fairness.algorithms.Ben import utils
from fairness.algorithms.Ben.margin import *
from fairness.algorithms.Algorithm import Algorithm

class SDBSVM(Algorithm):
   def __init__(self):
        Algorithm.__init__(self)
        self.name = "SDBSVM"

   def get_supported_data_types(self):
        return set(["numerical-binsensitive"])

   @arrayErrorBars(2)
   def statistics(self, train, test, protectedIndex, protectedValue, learner):
      #print("in statistics test-----------------------",test)
      h = learner(train, protectedIndex, protectedValue)
      #print("Computing error")
      error = labelError(test, h)
      #print("error on test set-------------------------", error)
      bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
      #print("Computing UBIF")
      ubif = individualFairness(train, learner, 0.2, passProtected=True)
      return error, bias, ubif

   def column_shift(self, data_df , class_attr):
      label_index = data_df.columns.get_loc(class_attr)
      cols=data_df.columns.tolist()
      cols=cols[:int(label_index)]+cols[int(label_index+1):len(cols)]+[class_attr]
      data_df=data_df[cols]
      data = data_df.values.tolist()
      return data

   def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
      # shifted labels to last column 
      train = self.column_shift(train_df, class_attr)
      unique_labels= list(OrderedDict().fromkeys(i[-1] for i in train))
      test = self.column_shift(test_df,class_attr)
      train_labels=[]
      train_data_points=[]
      test_labels=[]
      test_data_points=[]
      test = test_df.values.tolist()
      for datapoints in train:
         if (int(datapoints[-1]) != positive_class_val):
            datapoints[-1] = 0
         train_labels.append(int(datapoints[-1]))
         train_data_points.append(tuple(datapoints[:-1]))
      train= list(zip(train_data_points,train_labels))
      for datapoints in test:
         if (int(datapoints[-1]) != positive_class_val):
            datapoints[-1] = 0
         test_labels.append(int(datapoints[-1]))
         test_data_points.append(tuple(datapoints[:-1]))

      test= list(zip(test_data_points,test_labels))
      protectedIndex = train_df.columns.get_loc(sensitive_attrs[0])
      protectedValue = privileged_vals[0]
     
      prediction_ =self.runAll(train,test, protectedIndex, protectedValue)
      prediction_labels = list(set(prediction_))
      # change prediction back to original
      prediction=[]
      for item in prediction_:
         if (item != positive_class_val):
            if(unique_labels[0]!= positive_class_val):
               item = unique_labels[0]
            else:
               item = unique_labels[1]
         prediction.append(int(item))
      return  prediction, []
   
   def lrLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = lrSKLMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      #print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)
   
   def svmLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = svmRBFMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      #print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)


   def svmLinearLearner(self, train, protectedIndex, protectedValue):
      marginAnalyzer = svmLinearMarginAnalyzer(train, protectedIndex, protectedValue)
      shift = marginAnalyzer.optimalShift()
      #print('best shift is: %r' % (shift,))
      return marginAnalyzer.conditionalShiftClassifier(shift)

   @errorBars(10)
   def indFairnessStats(self, train, learner):
      #print("Computing UBIF")
      ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
      #print("UBIF:", ubif)
      return ubif

   def runAll(self,train, test, protectedIndex, protectedValue):
      #print("Shifted Decision Boundary Relabeling")
      #dataset = test+train
      experiments = [
      ('SVM', self.svmLinearLearner),
      ('SVM', self.svmLearner)
      #('lrLearner', self.lrLearner),
         ]

      for (learnerName, learner) in experiments:
         #print("%s" % (learnerName), flush=True)
         return experimentCrossValidate(train, test, learner, 2, self.statistics, protectedIndex, protectedValue)
      
