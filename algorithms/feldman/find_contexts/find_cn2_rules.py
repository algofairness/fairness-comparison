import Orange
import csv

def CN2_learner(orig_data, output_dir, beam_width, min_covered_examples, max_rule_length, influence_scores):
	# format data for classification
        original_data = Orange.data.Table.from_file(orig_data)
        # set the learner
        learner = Orange.classification.rules.CN2Learner()
        # set the number of solution steams considered at one time
        learner.rule_finder.search_algorithm.beam_width = beam_width
        # continuous value space is constrained to reduce computation time
        learner.rule_finder.search_strategy.constrain_continuous = True
        # set the minimum number of examples a found rule must cover to be considered
        learner.rule_finder.general_validator.min_covered_examples = min_covered_examples
        # set the maximum number of selectors (conditions) found rules may combine
        learner.rule_finder.general_validator.max_rule_length = max_rule_length
	
        # produce rules from unrepaired data
        classifier = learner(original_data)

        # write rules to file
	RULES_FILE = "{}/rules.csv".format(output_dir)
        with open(RULES_FILE, 'w') as csvfile:
                rules = csv.writer(csvfile)
                # Create rules file from repaired data
                rules.writerow(["Label","Rules","Quality","Score"])
                for rule_num, rule in enumerate(classifier_orig.rule_list):
			# calculate influence score
			domain = rule.domain.attributes
			selectors = rule.selectors
			score = sum([float(scores[domain[s.column].name]) for s in selectors])
                        # write rule details to file
			rules.writerow([rule_num, str(rule).strip(' '), rule.quality, score])
	return RULES_FILE
	

