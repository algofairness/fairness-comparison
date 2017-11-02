import argparse
import csv

from repairers.GeneralRepairer import Repairer

parser = argparse.ArgumentParser(description="Repair a CSV file.")
parser.add_argument("input_csv", type=str,
                   help="The file on which to perform the repair.")
parser.add_argument("output_csv", type=str,
                   help="The name to be used for the repaired CSV file.")
parser.add_argument("repair_level", type=float,
                   help="The level at which the repair should be performed between 0.0 and 1.0.")

parser.add_argument("-p", "--protected", type=str, nargs="+", required=True)
parser.add_argument("-i", "--ignored", type=str, nargs="+")

args = parser.parse_args()

with open(args.input_csv) as f:
  data = [line for line in csv.reader(f)]
  headers = data.pop(0)
  cols = [[row[i] for row in data] for i,col in enumerate(headers)]

  # Convert integer features to integers and float features to floats.
  for i, col in enumerate(cols):
    try:
      cols[i] = map(int, col)
    except ValueError:
      try:
        cols[i] = map(float, col)
      except ValueError:
        pass

  data = [[col[j] for col in cols] for j in xrange(len(data))]

# Calculte the indices to repair by and to ignore.
for protected in args.protected:
  try:
    index_to_repair = headers.index(protected)
  except ValueError as e:
    raise Exception("Response header '{}' was not found in the following headers: {}".format(protected, headers))

  try:
    ignored_features = [headers.index(feature) for feature in args.ignored] if args.ignored else []
  except ValueError as e:
    raise Exception("One or more ignored-features were not found in the headers: {}".format(headers))

  repairer = Repairer(data, index_to_repair,
                      args.repair_level, features_to_ignore=ignored_features)


  # Repair the input data and write it to a CSV.
  data = repairer.repair(data)

with open(args.output_csv, "wb") as f:
  writer = csv.writer(f)
  writer.writerow(headers)
  for row in data:
    writer.writerow(row)
