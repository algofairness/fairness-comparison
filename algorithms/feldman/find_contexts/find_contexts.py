import argparse
import time
from convert_and_merge_files import convert_to_tab, merge
from extract_influence_scores import extract_feature_influences 
from find_cn2_rules import CN2_learner
from expand_and_find_context import expand_and_find_contexts_of_influence

parser = argparse.ArgumentParser(description="Find contexts of influence in a dataset.")
parser.add_argument("original_csv", type=str,
                   help="The file for which contexts of influence are discovered.")
parser.add_argument("obscured_csv", type=str,
                   help="The file created by obscuring the original file.")
parser.add_argument("predicted_outcomes_csv", type=str,
                   help="Predicted response labels for the data.")
parser.add_argument("summary_file", type=str,
                   help="Summary file created by obscuring process.")
parser.add_argument("feature_data_file", type=str,
                   help="File that contains the types and/or domains of each feature")
parser.add_argument("output_txt", type=str,
                   help="The name to be used for the file that contains the results.")
parser.add_argument("removed_attr", type=str,
                   help="Name of the feature which the data is obscured with respect to")
parser.add_argument("beam_width", type=int, default=10,
                   help="The number of solution streams considered at one time.")
parser.add_argument("max_rule_length", type=int, default=5,
                   help="The maximum number of conditions that found rules may combine.")
parser.add_argument("min_covered_examples", type=int, default=1,
                   help="minimum number of examples a found rule must cover to be considered.")
args = parser.parse_args()

# Create directory to dump results
results_dir = "../ouput"
output_dir = "{}/{}".format(results_dir, time())

# Parse summary file for influence scores and generate obscured tag
influence_scores, obscured_tag = extract_feature_influences(args.summary_file, args.removed_attr)

# Convert original csv file to tab-separated
#	and create a new tab-separated file by merging original and obscured files
original_tab = convert_to_tab(args.original_csv, args.feature_data_file, output_dir)
merged_csv = merge(args.original_file, args.obscured_file, obscured_tag, output_dir)


# Parse summary file for influence scores and generate obscured tag
influence_scores, obscured_tag = extract_feature_influences(args.summary_file, args.removed_attr)

# Generate rule list for the original data using the CN2 algorithm
rulesfile = CN2_learner(original_tab, output_dir, args.beam_width, args.min_covered_examples, 
			args.max_rule_length, influence_scores)

# Generate fully expanded rule list, store best expanded rule for each of the original rules,
#	and return contexts of discrimination
contexts_of_influence = expand_and_find_contexts_of_influence(args.original_csv, merged_csv, 
							      rulesfile, influence_scores, 
							      obscured_tag, output_dir)

# Add experiment information to summary file

