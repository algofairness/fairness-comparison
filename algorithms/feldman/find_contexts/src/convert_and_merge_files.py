import csv

def convert_to_tab(csvfile, feature_data_file, output_dir):
	with open(csvfile, 'rt') as f:
		reader = csv.reader(f)
		# open tab-separated file
		tabfile = "{}/original.tab".format(output_dir)
		writer = open(tabfile, 'w')
		
		# write tab-separated header to tabfile
		header = next(reader)
		writer.write('\t'.join(header)+'\n')

		# extract feature data
		feature_data = csv.reader(open(feature_data_file, 'rt'))
		next(feature_data) # ignore header
		feature_domains = next(feature_data)
		metadata = next(feature_data)
		
		# write tab-separated feature data
		writer.write('\t'.join(feature_domains)+'\n')
		writer.write('\t'.join(metadata)+'\n')

		# convert csv rows to tab-separated and write
		for row in reader:
			writer.write('\t'.join(row)+'\n')

		return tabfile

def merge(orig_file, obscured_file, obscured_tag, output_dir):
	orig_reader = csv.reader(open(orig_file, 'rt'))
	obscured_reader = csv.reader(open(obscured_file, 'rt'))

	merged_csv = "{}/merged.csv".format(output_dir)
	merged_output = csv.writer(open(merged_csv, 'w'))

	# Created merged header by adding  obscured tag to feature names in obscured file 
	#	and merging with original feature names 
	orig_header = next(orig_reader)
	obscured_header = [attr+obscured_tag for attr in next(obscured_reader)]
	merged_header = [None]*(len(orig_header)+len(obscured_header))
	merged_header[::2] = orig_header
	merged_header[1::2] = obscured_header

	merged_output.writerow(merged_header)

	# merge all data and write
	for orig_row in orig_reader:
		obscured_row = next(obscured_reader)
		merged_row = [None]*(len(orig_row)+len(obscured_row))
		merged_row[::2] = orig_row
		merged_row[1::2] = obscured_row
		merged_output.writerow(merged_row)

	return merged_csv

import os, shutil, filecmp
def test():
	TMP_DIR = "tmp"
	if not os.path.exists(TMP_DIR):
		os.mkdir(TMP_DIR)
	
	test_csv_contents = [[["ColA","ColB","ColC"],
                          ["A","B","C"]],
                         [["ColA","ColB","ColC"],
                          ["A'","B'","C'"]],
                         [["ColA","ColB","ColC"],
                          ["d","d","d"],
                          ["","","class"]],
                         [["ColA","ColA-tag","ColB","ColB-tag","ColC","ColC-tag"],
                          ["A","A'","B","B'","C","C'"]]]
	test_csv_filenames = [TMP_DIR + "/test_file_original",
                      TMP_DIR + "/test_file_obscured",
                      TMP_DIR + "/test_file_feature_data",
                      TMP_DIR + "/test_merged"]

	test_tab_contents = "ColA\tColB\tColC\nd\td\td\n\t\tclass\nA\tB\tC\n"
	test_tab_filename = TMP_DIR + "/test_converted"
	
	for i, filename in enumerate(test_csv_filenames):
		with open(filename, 'w') as csvf:
			f = csv.writer(csvf)
			for row in test_csv_contents[i]:
				f.writerow(row)

	with open(test_tab_filename, 'w') as tabf:
		tabf.write(test_tab_contents)

	converted = convert_to_tab(test_csv_filenames[0], test_csv_filenames[2], TMP_DIR)
	assert(filecmp.cmp(converted, test_tab_filename))

	tag = "-tag"
	merged = merge(test_csv_filenames[0],test_csv_filenames[1],tag, TMP_DIR)
	assert(filecmp.cmp(merged, test_csv_filenames[3]))

	shutil.rmtree(TMP_DIR)

if __name__=="__main__": test()	
