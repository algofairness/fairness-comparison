import os, sys
from subprocess import call
# Open a file
path = "00RESULT/"
dirs = os.listdir(path)

for file in dirs:
    x = "python fai_bin_bin.py 00RESULT/"+file+" "+"00CALCULATIONS/"+file
    print x
    os.system(x)
