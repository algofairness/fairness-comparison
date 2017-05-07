import decimal
import csv
import random
import string
import datetime
import os
#from enums import repair_type
from itertools import product
from collections import defaultdict

def generate_file_id():
    # Pseudorandom 25-char + timestamp string to serve as the internal file ID
    file_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits*2) for x in range(25))
    file_id += datetime.datetime.now().isoformat()[:19].replace('-','').replace(':','') # Add timestamp
    return file_id

# Identifier columns contain information that should be irrelevant to a classfier. These columns will be ignored
#	Examples: applicant IDs, phone numbers
# Proected columns contain information that should not be used in a classifier
# 	Examples: race, gender, religion
# Statify columns contain values of groups which should be treated equally by a classifier. There must be a least one Statify column per data set
# Class columns contain the outcomes assigned to applicants. There must be exactly one Class column per data set
#	The last column in any data set
def setup_and_call_repair(quickname, filepath, repair_directory, identifier_col_names, protected_col_names,
				stratify_col_names, repair_amount, requested_repair_type):

    file_id = generate_file_id()
    orig_file = open(filepath, 'rb')
    file_reader = csv.reader(orig_file)

    # Extract column names from the given file
    col_names = file_reader.next()

    # Get column type information
    col_types = ["Y"]*(len(col_names) - 1) + ["C"]
    for i, col in enumerate(col_names):
        if 	col in identifier_col_names:	col_types[i] = "I"
        elif	col in protected_col_names:	col_types[i] = "X"
        else: continue

    col_type_dict =  {col_name: col_type for col_name, col_type in zip(col_names, col_types)}
    Y_col_names =	    filter(lambda x: col_type_dict[x] == "Y", col_names)
    not_I_col_names =	filter(lambda x: col_type_dict[x] != "I", col_names)

    # To prevent potential perils with user-provided column names, map them to safe column names
    safe_col_names = {col_name: "col_"+str(i) for i, col_name in enumerate(col_names)}
    safe_stratify_cols = [safe_col_names[col] for col in stratify_col_names]

    # Extract column values for each attribute in data
    # Begin by initializing keys and values in dictionary
    orig_data = {safe_col_names[col_name]: [] for col_name in col_names}
    # Populate each attribute with its column values
    for row in file_reader:
        for i, col_name in enumerate(col_names):
            if col_name in Y_col_names:	orig_data[safe_col_names[col_name]].append(float(row[i]))
	    else: 			orig_data[safe_col_names[col_name]].append(row[i])


    # Create unique value structures:
    # When performing repairs, we choose median values. If repair is partial, then values will
    # be modified to some intermediate value between the original and the median value. However,
    # the partially repaired value will only be chosen out of values that exist in the data set.
    # This prevents choosing values that might not make any sense in the data's context.
    # To do this, for each column, we need to sort all unique values and create two data structures:
    # a list of values, and a dict mapping values to their positions in that list. Example:
    #     There are unique_col_vals[col] = [1, 2, 5, 7, 10, 14, 20] in the column. A value 2 must be
    #     repaired to 14, but the user requests that data only be repaired by 50%. We do this by
    #     finding the value at the right index:
    #     index_lookup[col][2] = 1; index_lookup[col][14] = 5; this tells us that
    #     unique_col_vals[col][3] = 7 is 50% of the way from 2 to 14.
    unique_col_vals = {}
    index_lookup = {}
    for col_name in not_I_col_names:
    	col_values = orig_data[safe_col_names[col_name]]
        # extract unique values from column and sort
    	col_values = sorted(list(set(col_values)))
    	unique_col_vals[safe_col_names[col_name]] = col_values
    	# look up a value, get its position
    	index_lookup[safe_col_names[col_name]] = {col_values[i]: i for i in range(len(col_values))}


    # Make a list of unique values per each stratified column.
    # Then make a list of combinations of stratified groups. Example: race and gender cols are stratified:
    # [(white, female), (white, male), (black, female), (black, male)]
    # The combinations are tuples because they can be hashed and used as dictionary keys.
    # From these, find the sizes of these groups.
    unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
    all_stratified_groups = list(product(*unique_stratify_values))
    # look up a stratified group, and get a list of indices corresponding to that group in the data
    stratified_group_indices = defaultdict(list)
    # Find the sizes of each combination of stratified groups in the data
    sizes = {group: 0 for group in all_stratified_groups}
    for i in range(len(orig_data[safe_stratify_cols[0]])):
    	group = tuple(orig_data[col][i] for col in safe_stratify_cols)
    	stratified_group_indices[group].append(i)
    	sizes[group] += 1

    # Don't consider groups not present in data (size 0)
    all_stratified_groups = filter(lambda x: sizes[x], all_stratified_groups)

    # Separate data by stratified group to perform repair on each Y column's values given that their
    # corresponding protected attribute is a particular stratified group. We need to keep track of each Y column's
    # values corresponding to each particular stratified group, as well as each value's index, so that when we
    # repair the data, we can modify the correct value in the original data. Example: Supposing there is a
    # Y column, "Score1", in which the 3rd and 5th scores, 70 and 90 respectively, belonged to black women,
    # the data structure would look like: {("Black", "Woman"): {Score1: [(70,2),(90,4)]}}
    stratified_group_data = {group: {} for group in all_stratified_groups}
    for group in all_stratified_groups:
    	for col_name in orig_data:
            stratified_col_values = sorted([(orig_data[col_name][i], i) for i in stratified_group_indices[group]], key=lambda vals: vals[0])
            stratified_group_data[group][col_name] = stratified_col_values

    # Find the combination with the fewest data points. This will determine what the quantiles are.
    num_quantiles = min(filter(lambda x: x, sizes.values()))

    # Repair Data and retrieve the results
    repaired_data = perform_repair(quickname, repair_directory, col_names, col_type_dict, safe_col_names, all_stratified_groups,
					sizes, stratified_group_indices, stratified_group_data, repair_amount, num_quantiles, orig_data,
					index_lookup, unique_col_vals, requested_repair_type)
    print "Done with " + quickname + " " + str(repair_amount) + '\n' + "Preparing data now..."

    # Organize repaired data into a csv_file
    #repair_dir = repair_directory+"/Repaired_Data_Files/"+file_id+"_"+quickname+"_"+str(repair_amount)+"/"
    repair_dir = "data/adult/Repaired_Data_Files/"

    #os.mkdir(repair_dir)

    # return to beginning of original file
    orig_file.seek(0)
    file_reader.next()
    with open(repair_dir+"Fixed_Adult_1.csv", 'wb') as fixed_data:
    	writer = csv.writer(fixed_data, delimiter = ',')
    	# write original column names to file with X and I cols
    	repaired_col_names = filter(lambda x: col_type_dict[x] in 'YC', col_names)
    	writer.writerow(repaired_col_names)
    	for row_number in range(len(repaired_data["col_0"])):
            repaired_row = [repaired_data[safe_col_names[col_name]][row_number] for col_name in repaired_col_names]
            writer.writerow(repaired_row)

    print "Repair Process Complete. Repaired Data can be found at: ", repair_dir
    return repair_dir+"Fixed_Data_9.csv"


def perform_repair(quickname, repair_directory, col_names, col_type_dict, safe_col_names, all_stratified_groups,
			sizes, stratified_group_indices, stratified_group_data, repair_amount, num_quantiles, orig_data,
			index_lookup, unique_col_vals, requested_repair_type):
    quantile_unit = 1.0/num_quantiles
    repaired_data = orig_data

    #with open(repair_directory+"/"+str(repair_amount)+".txt", 'a') as thelog:
    with open(repair_directory+"/log.txt", 'a') as thelog:
        thelog.write("Quantile Unit: {}; Num Quantiles: {}".format(quantile_unit, num_quantiles))

    for col_name in filter(lambda x: col_type_dict[x] == "Y", col_names):
        with open(repair_directory+"/"+str(repair_amount)+".txt", 'a') as thelog:
            thelog.write(str(col_name) + 'n')

   	# which bucket value we're repairing
        group_offsets = {group: 0 for group in all_stratified_groups}
        col = orig_data[safe_col_names[col_name]]
        for quantile in range(num_quantiles):
            values_at_quantile = []
            indices_per_group = {}
            for group in all_stratified_groups:
                offset = int(round(group_offsets[group]*sizes[group]))
                number_to_get = int(round((group_offsets[group] + quantile_unit)*sizes[group]) - offset)
                group_offsets[group] += quantile_unit

            	# get data at this quantile from this Y column such that stratified X = group
            	group_data_at_col = stratified_group_data[group][safe_col_names[col_name]]
		# (val, index) -> tuple

                indices_per_group[group] = [x[1] for x in group_data_at_col[offset:offset+number_to_get]]
#            	values = [float(x[0]) if isinstance(x[0], decimal.Decimal) else x[0] for x in group_data_at_col[offset:offset+number_to_get]]

                values =  [x[0] for x in group_data_at_col[offset:offset+number_to_get]]
            	# Find this group's median value at this quantile
                values_at_quantile.append(sorted([float(x) for x in values])[len(values)/2])

            # Find the median value of all groups at this quantile (chosen from each group's medians)
            median = sorted(values_at_quantile)[len(values_at_quantile)/2]
            median_val_pos = index_lookup[safe_col_names[col_name]][median]

            # Update values to repair the dataset!
            for group in all_stratified_groups:
                for index in indices_per_group[group]:
                    original_value = col[index]

                    """
                    #Evan is changing this on 4/14
                    #if requested_repair_type == repair_type.COMBINATORIAL:"""

                    if requested_repair_type == "0":
                        current_val_pos = index_lookup[safe_col_names[col_name]][original_value]
                        distance = median_val_pos - current_val_pos # distance between indices
                        distance_to_repair = int(round(distance * repair_amount))
                        index_of_repair_value = current_val_pos + distance_to_repair
                        repaired_value = unique_col_vals[safe_col_names[col_name]][index_of_repair_value]
                    else:
                        print col_name
                        repaired_value = (1 - repair_amount)*original_value+repair_amount*median

                    # Update data to repaired valued
                    repaired_data[safe_col_names[col_name]][index] = repaired_value

    # Remove all I and X cols
    for col_name in filter(lambda x: col_type_dict[x] in 'IX', col_names):
    	repaired_data.pop(col_name, None)

    return repaired_data

# setup_and_call_repair(quickname, filepath, repair_directory, identifier_col_names, protected_col_names,
# 				stratify_col_names, repair_amount, requested_repair_type):
#age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income-per-year


# Identifier columns contain information that should be irrelevant to a classfier. These columns will be ignored
#	Examples: applicant IDs, phone numbers
# Proected columns contain information that should not be used in a classifier
# 	Examples: race, gender, religion
# Statify columns contain values of groups which should be treated equally by a classifier. There must be a least one Statify column per data set
# Class columns contain the outcomes assigned to applicants. There must be exactly one Class column per data set
#	The last column in any data set


identifier_col_names = []
stratify_col_names = ["sex"]
protected_col_names = []
repair_values = 1.0
requested_repair_type = "0"
print "Call repair"
setup_and_call_repair("name", "data/adult/adult-all-numerical-converted.csv", "data/adult", identifier_col_names, protected_col_names, stratify_col_names, repair_values, requested_repair_type)
print "Done"


# def repair_script_test():
#     quickname ="Ricci_Test"
#     filepath = "../media/RicciDataMod.csv"
#     repair_directory =  "../Script_Test_Kav"
#     identifier_col_names = ["Position"]
#     protected_col_names = ["Race"]
#     stratify_col_names = ["Race"]
#     #repair_values = [i*1.0/10 for i in range(11)]
#     repair_values = [1.0]
#     requested_repair_type = repair_type.COMBINATORIAL
#
#     for repair_amount in repair_values:
#         repaired_data = setup_and_call_repair(quickname, filepath, repair_directory, identifier_col_names, protected_col_names,
# 						stratify_col_names, repair_amount, requested_repair_type)
#
#         correct_data = repair_directory+"/Ricci_Fixed/ricci_test_mod" + str(repair_amount) + "/Fixed Data.csv"
#
#         reader1 = csv.reader(open(repaired_data, 'rb'))
#         reader2 = csv.reader(open(correct_data, 'rb'))
#
#         # Skip the header!
#         reader1.next()
#         reader2.next()
#         for i, repaired_row in enumerate(reader1):
#             correct_row = reader2.next()
#             # change strings to floats to compare the two
#             repaired = [float(x) for x in repaired_row]
#             correct = [float(x) for x in correct_row]
#             if repaired != correct:
#                 print "Oops Repair Script Incorrect!", i
#                 print "Got:      ", repaired
#                 print "Expected: ", correct
#
#
#         print "Everything looks fine for repair value " +str(repair_amount)
#


if __name__ == "__main__": repair_script_test()
