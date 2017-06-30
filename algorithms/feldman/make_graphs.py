"""
Development script for converting audit files into graphs.
"""

from audit_reading import graph_audits, graph_audit, rank_audit_files
from measurements import BCR, accuracy

from os import listdir
from os.path import isfile, join

import sys


def audit_directory(directory):
  only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
  audits = ["{}/{}".format(directory, f) for f in only_files if f[-6:]==".audit"]

  measurers = [accuracy, BCR]
  for audit in audits:
    audit_image_filename = audit + ".png"
    graph_audit(audit, measurers, audit_image_filename)

  for measurer in measurers:
    ranked_graph_filename = "{}/{}.png".format(directory, measurer.__name__)
    graph_audits(audits, measurer, ranked_graph_filename)
    ranks = rank_audit_files(audits, measurer)
    print measurer.__name__, ranks


if __name__=="__main__":
  try:
    directory = sys.argv[1]
    audit_directory(directory)
  except:
    print "proper usage: make_graphs.py <directory/with/audits>"

