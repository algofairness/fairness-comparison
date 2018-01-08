# minimal polyfill for commands in python3

import subprocess

def getoutput(cmd):
    return subprocess.run(cmd, shell=True, encoding='utf-8', stdout=subprocess.PIPE).stdout

