import os, sys
import urllib2


def CheckData(dataset):
	path = '../../data'
	print('Ensuring you have the ' + str(dataset) + ' dataset files')
	
	if dataset == 'adult' or 'german':
		path = '../../data/' + dataset
		filename = dataset + '_data'
		
		if filename not in os.listdir('path'):
			print "'%s' not found! Downloading from UCI Archive..." % fname
			addr = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/%s" % fname
			response = urllib2.urlopen(addr)
			data = response.read()
			fileOut = open(fname, "w")
			fileOut.write(data)
			fileOut.close()
			print "'%s' download and saved locally.." % fname
    else:
        print "File found in current directory.."
	
