import os

# Adult
print "Running Adult repair..."
os.system("python repair.py test_data/adult.csv results/adult_repaired_feldman.csv 1 -p sex -i race")
print "results saved in feldman/results/adult_repaired_feldman.csv"
