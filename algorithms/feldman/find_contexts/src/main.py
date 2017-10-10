import argparse
import time
import os
from distutils.util import strtobool
from convert_and_merge_files import convert_to_tab, merge
from extract_influence_scores import extract_feature_influences 
from find_cn2_rules import CN2_learner
from expand_and_find_contexts import expand_and_find_contexts

parser = argparse.ArgumentParser(description="Find contexts of influence in a dataset.")
parser.add_argument("original_csv", type=str,
                   help="The file for which contexts of influence are discovered.")
parser.add_argument("obscured_csv", type=str,
                   help="The file created by obscuring the original file.")
parser.add_argument("summary_file", type=str,
                   help="Summary file created by obscuring process.")
parser.add_argument("feature_data_file", type=str,
                   help="File that contains the types and/or domains of each feature")
parser.add_argument("removed_attr", type=str,
                   help="Name of the feature which the data is obscured with respect to")
parser.add_argument("beam_width", type=int, default=10,
                   help="The number of solution streams considered at one time.")
parser.add_argument("min_covered_examples", type=int, default=1,
                   help="minimum number of examples a found rule must cover to be considered.")
parser.add_argument("max_rule_length", type=int, default=5,
                   help="The maximum number of conditions that found rules may combine.")
parser.add_argument("by_original", type=strtobool, default=True,
                   help="Consider best expanded rule within epsilon of original quality (True) or best quality of expanded rules (False)")
parser.add_argument("output", type=str, default=None,
                   help="Name of output directory")
args = parser.parse_args()

# Create directory to dump results
results_dir = "../output"
output_dir = args.output if args.output else "{}/{}".format(results_dir, time.time())

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Parse summary file for influence scores and generate obscured tag
influence_scores, obscured_tag = extract_feature_influences(args.summary_file, args.removed_attr)

# Convert original csv file to tab-separated
#	and create a new tab-separated file by merging original and obscured files
original_tab = convert_to_tab(args.original_csv, args.feature_data_file, output_dir)
merged_csv = merge(args.original_csv, args.obscured_csv, obscured_tag, output_dir)

# Generate rule list for the original data using the CN2 algorithm
rulesfile = CN2_learner(original_tab, output_dir, args.beam_width, args.min_covered_examples, args.max_rule_length, influence_scores)

# Generate fully expanded rule list, store best expanded rule for each of the original rules,
#	and return contexts of discrimination
contexts_of_influence = expand_and_find_contexts(args.original_csv, args.obscured_csv, merged_csv, rulesfile, influence_scores, obscured_tag,output_dir, args.by_original)

# Write results to summary file:
summary = "{}/summary.txt".format(output_dir)
summary_file = open(summary, 'w')

summary_file.write("CN2 Settings Used:\n")
summary_file.write("rules found for {}\n".format(args.original_csv))
summary_file.write("beam_width: {}\n".format(args.beam_width))
summary_file.write("min_covered_examples: {}\n".format(args.min_covered_examples))
summary_file.write("max_rule_length: {}\n\n".format(args.max_rule_length))

summary_file.write("Contexts of influence found:\n")
for outcome in contexts_of_influence:
	list_of_contexts = contexts_of_influence[outcome]
	contexts = " OR \n".join([", ".join(context) for context in list_of_contexts])
	summary_file.write(outcome +': \n')
	summary_file.write(contexts + '\n\n')

print("Summary of Experiment written to {}".format(summary))
