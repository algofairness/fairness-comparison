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
		writer.write('\t'.join(meatdata)+'\n')

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
                repaired_row = next(repaired_reader)
                merged_row = [None]*(len(orig_row)+len(repaired_row))
                merged_row[::2] = orig_row
                merged_row[1::2] = repaired_row
                merged_output.writerow(merged_row)

	return merged_csv
