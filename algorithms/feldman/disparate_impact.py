# Parameters: feature_to_repair, response are both lists, groups and outcomes are tuples
# DI = Pr(C=1|X=0)/Pr(C=1|X=1)
def disparate_impact(list_of_triples, nonprotected_group, protected_group):
	# Assert len(feature_to_repair) == len(response), and groups and outcomes are tuples
	#SPECIFIC TO RECITIVISM DATA!!
	print "Calculating DI for: ", protected_group
	feature_to_repair = []
	groups = {}
	response = []
	for triple in list_of_triples:
		#Actually, find for UKNOWN too if triple[0] != missing_group:
		groups[triple[0]] = 0
	for triple in list_of_triples:
		groups[triple[0]] += 1
		feature_to_repair.append(triple[0])
		response.append(int(triple[2]))
	group_list =[]
	for group in groups:
		group_list.append(group)
	total = len(feature_to_repair)
	#group_x white!
	group_x = [0]*total
	#group_y is not_white!
	group_y = [0]*total
	group_x_and_a = [0]*total	
	group_y_and_a = [0]*total	
	for i, group in enumerate(feature_to_repair):
		if group == nonprotected_group:
			group_x[i] = 1
			if response[i] == 0:
				group_x_and_a[i] = 1
		if group == protected_group:
			group_y[i] = 1 
			if response[i] == 0:
				group_y_and_a[i] = 1
	prob_x = sum(group_x) / float(total)
	prob_y = sum(group_y) / float(total)
	prob_x_and_a = sum(group_x_and_a) / float(total)
	prob_y_and_a = sum(group_y_and_a) / float(total)
	di = 0 
	if prob_x==0:
		prob_a_given_x=1
	else:
		prob_a_given_x = prob_x_and_a / prob_x
	if prob_y==0:
		prob_a_given_y=1
	else:
		prob_a_given_y = prob_y_and_a / prob_y
	if prob_a_given_x ==0:
		di= 1
	else:
		di = prob_a_given_y/prob_a_given_x
	print "Stats:", prob_x, prob_y, prob_x_and_a, prob_y_and_a, prob_a_given_x,prob_a_given_y,   di
	return di


def disparate_impact2(feature_to_repair, response, groups, outcomes):
	# Assert len(feature_to_repair) == len(response), and groups and outcomes are tuples
	
	total = len(feature_to_repair)
	group_x = [0]*total
	group_y = [0]*total
	group_x_and_a = [0]*total	
	group_y_and_a = [0]*total	
	for i, group in enumerate(feature_to_repair):
		if group == groups[0]:
			group_x[i] = 1
			if response[i] == outcomes[0]:
				group_x_and_a[i] = 1
		elif group == groups[1]:
			group_y[i] = 1 
			if response[i] == outcomes[0]:
				group_y_and_a[i] = 1
	prob_x = sum(group_x) / float(total)
	prob_y = sum(group_y) / float(total)
	prob_x_and_a = sum(group_x_and_a) / float(total)
	prob_y_and_a = sum(group_y_and_a) / float(total)
	prob_a_given_x = prob_x_and_a / prob_x
	prob_a_given_y = prob_y_and_a / prob_y
	di = prob_a_given_y / prob_a_given_x
	return prob_a_given_y/prob_a_given_x

def test():
	feature_to_repair = ['W','W','B','B','W']
	response = [0,1,1,0,1]
	groups = ('W','B')
	outcomes = (1,0)
	di = disparate_impact2(feature_to_repair, response, groups, outcomes)
	di = round(di, 2)
	print "Disparate Impact correct?", di == 0.75 

if __name__== "__main__":
  test()
