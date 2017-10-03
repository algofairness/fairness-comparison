import ast

def extract_feature_influences(summary_file, removed_attr):
	f = open(summary_file, 'r')
	influence_scores_dict = {}
	obscured_tag = "-no{}".format(removed_attr) 
	# find line in summary file that corresponds to influence score per feature
	for line in f:
		if line.startswith('Ranked Features by accuracy:'):
			# convert the data to a list of tuples 
			influence_scores_data = ast.literal_eval(line.split(':')[1][1:])
	
			# Each feature's influence is stored in a dictionary
			for element in influence_scores_data:
				influence_scores_dict[element[0]] = float(element[1])
				influence_scores_dict[element[0]+obscured_tag] = 0.0

	return influence_scores_dict, obscured_tag
