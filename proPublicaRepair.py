import os,sys
import numpy as np
from zafar_classifier import *
import csv
from load_compas_data import *
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints

import os, sys
from subprocess import call


X, y, x_control, feature_names = load_compas_data()
feature_names.append("recidivism")
X = X.tolist()
y = y.tolist()

for i in range(0, len(X)):
    X[i].append(y[i])

with open("cleaned_not_repaired_propublica", "wb") as f:
    writer = csv.writer(f)
    writer.writerow(feature_names)
    writer.writerows(X)
