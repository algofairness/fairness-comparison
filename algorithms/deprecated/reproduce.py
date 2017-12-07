import os

# Script to run all tests from Kamishima, Zafar, and Feldman
prevdir = os.getcwd()

# Kamishima tests
print "\n######################## Running Kamishima Tests ######################## \n "
os.chdir("kamishima")
os.system("bash test-all-task.sh")
print "results saved in kamishima/00RESULT"
os.chdir(prevdir)

# Feldman tests
print "\n######################## Running Feldman Tests ########################## \n "
os.chdir("feldman")
os.system("python run_all.py")
os.chdir(prevdir)

# Zafar tests
print "\n######################## Running Zafar Tests ############################ \n "
os.chdir("zafar/disparate_impact/adult_data_demo")
os.system("python demo_constraints.py")
os.chdir(prevdir)
os.chdir("zafar/disparate_mistreatment/propublica_compas_data_demo")
os.system("python demo_constraints.py")
os.chdir(prevdir)
