import operator
from collections import namedtuple, defaultdict
import csv
import ast
import os
import itertools


class Selector(namedtuple('Selector', 'var, op, val')):
	"""
	Define a single rule condition
	"""
	OPERATORS = {
		# discrete, nomial variables
		'==': operator.eq,
		'!=': operator.ne,
	
		# continous variables
		'<=': operator.le,
		'>=': operator.ge
	}

	def covers(self, x):
		return Selector.OPERATORS[self[1]](x[self[0]], self[2])	
	
	def __str__(self):
		return str(self[0])+str(self[1])+str(self[2])

class Rule:
	def __init__(self, selectors=None, outcome_var=None, outcome_val=None,
				 quality=None, influence_score=None, ID=None):
		self.selectors = selectors if selectors is not None else []
		self.outcome_var = outcome_var
		self.outcome_val = outcome_val
		self.quality = quality
		self.influence_score = influence_score
		self.ID = ID

	def calculate_influence_score(self, influence_scores):
		score = sum([influence_scores[s.var] for s in self.selectors])
		if self.influence_score is not None:
			assert(self.influence_score == score)
		else:
			self.influence_score = score

	def calculate_quality(self, data):
		covered_rows = []
		correct_covered_rows = []
		classes = set()
		for row in data:
			covers_row = True
			for selector in self.selectors:
				if not selector.covers(row):
					covers_row = False
			if covers_row:
				covered_rows.append(row)
				# check if outcome labels match
				if self.outcome_val == row[self.outcome_var]:
					correct_covered_rows.append(row)
			classes.add(row[self.outcome_var])

		num_covered = len(covered_rows)
		num_correct = len(correct_covered_rows)
		num_classes = len(classes)
		quality = (num_correct + 1.0) / (num_covered + num_classes)
		
		if self.quality is not None:
			assert(self.quality == quality)
		else:
			self.quality = quality

	def __str__(self):
		cond = " AND ".join([str(s) for s in self.selectors])
		outcome = "{}={}".format(self.outcome_var, self.outcome_val)
		return "IF {} THEN {}".format(cond, outcome)
		

"""
Format data 
"""
def get_data(datafile):
	reader = csv.DictReader(open(datafile, 'r'))
	data = []
	for row in reader:
		row_dict = {}
		for key in row:
			row_dict[key] = convert_ifnum(row[key])
		data.append(row_dict)
	return data

"""
Takes rules from the original rule list and converts them to Rule objects
"""
def is_int(string):
	try:
		int(string)
		return True
	except ValueError:
		return False

def is_float(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

def convert_ifnum(string):
	if is_int(string):
		return int(string)
	if is_float(string):
		return float(string)
	return string

def parse_rule(rule):
	antecedent, consequent = rule.split(" THEN ")
	outcome_var, outcome_val = consequent.split("=")
	
	selectors = antecedent[3:].split(" AND ") # ignore "IF " in antecedent
	Selectors = []

	for selector in selectors:
		if "==" in selector:
			op = '=='
		elif "!=" in selector:
			op = '!='
		elif ">=" in selector:
			op = ">="
		elif "<=" in selector:
			op = "<="
		elif "TRUE" in selector:
			continue
		else:
			print("Warning: unrecognized operation for " + selector)

		s_var, s_val = selector.split(op)
		s_val = convert_ifnum(s_val)
		Selectors.append(Selector(var=s_var, op=op, val=s_val))
		
	return Selectors, outcome_var, outcome_val
		

def get_rules_from_file(rulefile, data):
	reader = csv.DictReader(open(rulefile, "r"))
	rules = []
	for row in reader:
		rule_num = int(row["Label"])
		rule = row["Rules"]
		selectors, outcome_var, outcome_val = parse_rule(rule)
		influence_score = float(row["Score"])
		r = Rule(selectors=selectors, outcome_var=outcome_var, outcome_val=outcome_val, 
				 influence_score=influence_score, ID=rule_num)
		r.calculate_quality(data)
		rules.append(r)
	return rules

"""
Get the possible values that original features are mapped to in the obscured version.
"""

def get_orig_to_obscured_map(original_csv, obscured_csv):
	original_reader = csv.DictReader(open(original_csv, "r"))
	obscured_reader = csv.DictReader(open(obscured_csv, "r"))

	# dict mapping from attribute name to orig value to list of obscured values
	# "prior_count" -> { 3.0 -> [1.0, 2.0]}
	orig_to_obscured = {}

	# maps row num to the original value at an attribute
	# 2 -> {"prior_count" -> 1.0}
	rownum_to_origval = {}

	for i, rowdict in enumerate(original_reader):
		rownum_to_origval[i] = {}
		for attr in rowdict:
			attr_val = convert_ifnum(rowdict[attr])
			rownum_to_origval[i][attr] = attr_val

			if attr not in orig_to_obscured:
				orig_to_obscured[attr] = {}
			if attr_val not in orig_to_obscured[attr]:
				orig_to_obscured[attr][attr_val] = []

	for i, rowdict in enumerate(obscured_reader):
		for attr in rowdict:
			attr_val = convert_ifnum(rowdict[attr])

			if attr not in orig_to_obscured:
				print("Warning: can't find original attribute to match this obscured attribte:" + attr)
			orig_val = rownum_to_origval[i][attr]

			if attr_val not in orig_to_obscured[attr][orig_val]:
				orig_to_obscured[attr][orig_val].append(attr_val)
	return orig_to_obscured

"""
Take a single rule and expand it to all the possible versions based on the obscured values.
"""

def generate_obscured_selectors(selector, obscured_vals, obscured_tag):
	obscured_selectors = [selector]
	obscured_var = selector.var + obscured_tag
	op = selector.op
	for obscured_val in obscured_vals:
		obscured_selectors.append(Selector(var=obscured_var, op=op, val=obscured_val))
	return obscured_selectors

def get_expanded_from_rule(original_rule, orig_to_obscured, data, influence_scores, obscured_tag):
	expanded_rules = [original_rule]
	possible_obscured_selectors = []
	for selector in original_rule.selectors:
		obscured_vals = orig_to_obscured[selector.var][selector.val]
		obscured_selectors = generate_obscured_selectors(selector, obscured_vals, obscured_tag)
		possible_obscured_selectors.append(obscured_selectors)

	product = itertools.product(*possible_obscured_selectors)
	for combo in product:
		outcome_var = original_rule.outcome_var
		outcome_val = original_rule.outcome_val
		ID = original_rule.ID
		expanded_rule = Rule(selectors=combo, outcome_var=outcome_var,
							 outcome_val=outcome_val, ID=ID)
		expanded_rule.calculate_quality(data)
		expanded_rule.calculate_influence_score(influence_scores)
		expanded_rules.append(expanded_rule)
	return expanded_rules	

"""
Create a dictionary mapping original rule numbers to their expanded rule versions.
"""

def expand_rules(original_rules, orig_to_obscured, data, influence_scores, obscured_tag):
	expanded_rules_dict = {}
	for rule in original_rules:
		rule_num = rule.ID
		expanded_rules = get_expanded_from_rule(rule, orig_to_obscured, data,
												influence_scores, obscured_tag)
		expanded_rules_dict[rule_num] = expanded_rules
	return expanded_rules_dict

"""
For each rule number, find the rule that is within epsilon quality of the original rule
	for that number and has the lowest influence score
"""

def select_best_obscured_rules(original_rules, expanded_rules_dict, by_original, epsilon):
	best_rules = []
	for original_rule in original_rules:
		ID = original_rule.ID
		# find best rule with quality within epsilon of the original rule's quality
		if by_original:
			best_rule = original_rule
			best_influence_score = original_rule.influence_score
			for expanded_rule in expanded_rules_dict[ID]:
				if (expanded_rule.quality + epsilon >= original_rule.quality and
					expanded_rule.influence_score < best_influence_score):
					best_rule = expanded_rule
					best_influence_score = expanded_rule.influence_score
			best_rules.append(best_rule)
		# find best rule with quality within epsilon of highest quality rule of bunch
		else:
			best_rule = None
			best_influence_score = 0.0
			best_quality = 0.0
			# find rule with the highest quality
			for rule in expanded_rules_dict[ID]:
				best_influence_score = rule.influence_score if rule.quality > best_quality else best_influence_score
				best_rule = rule if rule.quality > best_quality else best_rule
				best_quality = rule.quality if rule.quality > best_quality else best_quality
			for expanded_rule in expanded_rules_dict[ID]:
				if (expanded_rule.quality + epsilon >= best_quality and 
					expanded_rule.influence_score < best_influence_score):
					best_rule = expanded_rule
					best_influence_score = expanded_rule.influence_score
			best_rules.append(best_rule)
	return best_rules

"""
Find contexts of influence
"""
def find_contexts_of_influence(rules, obscured_tag):
	contexts_dict = defaultdict(set)
	for rule in rules:
		non_obscured_conds = frozenset([str(s) for s in rule.selectors if obscured_tag not in s.var])
		outcome = rule.outcome_var+'='+rule.outcome_val
		if non_obscured_conds:
			contexts_dict[outcome].add(non_obscured_conds)
	return contexts_dict

"""
Expand rules and find all contexts of influence
"""
def expand_and_find_contexts(original_csv, obscured_csv, merged_csv, rulesfile,
					  influence_scores, obscured_tag, output_dir, by_original, epsilon):
	# Format data 
	data = get_data(merged_csv)

	# Convert rules form rule list into Rule objects
	original_rules = get_rules_from_file(rulesfile, data)
	
	# Get a mapping from the original data values to their obscured values
	orig_to_obscured = get_orig_to_obscured_map(original_csv, obscured_csv)
    
	print("Expanding rule list")
	# Expand each original rule and store them in a dictionary by the original rule's number
	expanded_rules_dict = expand_rules(original_rules, orig_to_obscured, data, influence_scores, obscured_tag)
	
	print("Finding best expanded rules")
	# Find the best expanded rule for each rule number
	best_rules = select_best_obscured_rules(original_rules, expanded_rules_dict, by_original, epsilon)

	# Find the contexts of influence for the best expanded rules
	contexts_of_influence = find_contexts_of_influence(best_rules, obscured_tag)
	
	# Write results to files 
	header = ["Label", "Rule", "Quality", "Influence"]

	print("Writing all expanded rules to file.")
	full_expanded_file = open("{}/full_expanded_rulelist.csv".format(output_dir), 'w')
	full_expanded_writer = csv.DictWriter(full_expanded_file, fieldnames=header)
	
	for rule_num in expanded_rules_dict:
		expanded_rules = expanded_rules_dict[rule_num]
		for rule in expanded_rules:
			rule_ID = rule.ID
			quality = rule.quality
			influence = rule.influence_score
			row_dict = {"Label": rule_ID, "Rule": str(rule), "Quality": quality, "Influence": influence}
			full_expanded_writer.writerow(row_dict)

	print("Writing best expanded rules results to file.")
	best_expanded_file = open("{}/best_expanded_rulelist.csv".format(output_dir), 'w')
	best_expanded_writer = csv.DictWriter(best_expanded_file, fieldnames=header)

	for rule in best_rules:
		rule_num = rule.ID
		quality = rule.quality
		influence = rule.influence_score
		row_dict = {"Label": rule_num, "Rule": str(rule), "Quality": quality, "Influence": influence}
		best_expanded_writer.writerow(row_dict)

	print("Writing contexts of influence results to file.")
	contexts_file = open("{}/contexts_of_influence.txt".format(output_dir), 'w')
	for outcome in contexts_of_influence:
		list_of_contexts = contexts_of_influence[outcome]
		contexts = " OR \n".join([" AND ".join(context) for context in list_of_contexts])
		contexts_file.write(outcome +': \n')
		contexts_file.write(contexts + '\n\n')

	return contexts_of_influence

import shutil, filecmp
def test():
	TMP_DIR = "tmp"
	if not os.path.exists(TMP_DIR):
		os.mkdir(TMP_DIR)
	
	test_orig_contents = [["Col1","Col2","Class"],
                          ["A","X",1],
                          ["A","X",1],
                          ["A","Y",0],
                          ["B","X",0],
                          ["B","Y",0]]

	test_obscured_contents = [["Col1", "Col2","Class"],
                              ["a","x",1],
                              ["a","x",1],
                              ["a","y",0],
                              ["b","x",0],
                              ["b","y",0]]


	test_merged_contents = [["Col1","Col1-tag","Col2","Col2-tag","Class"],
                              ["A","a","X","x",1],
                              ["A","a","X","x",1],
                              ["A","a","Y","y",0],
                              ["B","b","X","x",0],
                              ["B","b","Y","y",0]] 

	test_rulesfile = [["Label","Rule","Quality","Score"],
                      [0,"IF Col1!=A THEN Class=0",0.75,0.15],
                      [1,"IF Col2==X THEN Class=1",0.75,0.15],
                      [2,"IF TRUE THEN Class=0",.67,0]]

	test_full_expanded_rulelist = [[0,"IF Col1!=A THEN Class=0",0.25,0.15],
                                   [0,"IF Col1!=A THEN Class=0",0.25,0.15],
                                   [0,"IF Col1-notag!=a THEN Class=0",0.25,0.0],
                                   [1,"IF Col2==X THEN Class=1",0.2,0.15],
                                   [1,"IF Col2==X THEN Class=1",0.2,0.15],
                                   [1,"IF Col2-notag==x THEN Class=1",0.2,0.0],
                                   [2,"IF  THEN Class=0",0.14285714285714285,0.0],
                                   [2,"IF  THEN Class=0",0.14285714285714285,0]]

	test_best_expanded_rulelist = [[0,"IF Col1-notag!=a THEN Class=0",0.25,0.0],
                                   [1,"IF Col2-notag==x THEN Class=1",0.2,0.0],
                                   [2,"IF  THEN Class=0,0.14285714285714285",0.0]]

	test_influence_scores = {"ColA":.15, "ColB":.15, "ColA-tag":0, "ColB-tag":0}
	obscured_tag = "-tag"
	outputdir = TMP_DIR
	epsilon = .1

	test_files = [(TMP_DIR + "/test_orig", test_orig_contents),
                  (TMP_DIR + "/test_obscured", test_obscured_contents), 
                  (TMP_DIR + "/test_merged", test_merged_contents), 
                  (TMP_DIR + "/test_rulesfile", tests_rulesfile),
                  (TMP_DIR + "/test_full_rules", test_full_expanded_rulelist), 
                  (TMP_DIR + "/test_best_rules", test_best_expanded_rulelist)]

	for test_file in test_files:
		with open(test_file[0], 'w') as csvf:
			f = csv.writer(csvf)
			for row in test_file[1]:
				f.writerow(row)

	contexts_of_influence = expand_and_find_contexts(original_csv, obscured_csv, merged_csv, rulesfile, influence_scores, obscured_tag, output_dir, by_original, epsilon)
	
	res_files = [(TMP_DIR+"/full_expanded_rulelist.csv"),
                 (TMP_DIR+"/best_expanded_rulelist.csv")]

	assert(filecmp.cmp(res_files[0], test_file[4][0]))
	assert(filecmp.cmp(res_files[1], test_file[5][0]))

	shutil.rmtree(TMP_DIR)
