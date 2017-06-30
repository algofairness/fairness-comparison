from splitters import split_by_percent
import csv

train_percentage = 2.0/3.0
#train_percentage = 0.8
print train_percentage

def load_data():
  filename = ""  # file not publicly available
  with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]
    headers = data.pop(0)

    correct_types = [str] * len(headers) # All categorical.

    for i, row in enumerate(data):
      for j, correct_type in enumerate(correct_types):
        data[i][j] = correct_type(row[j])

    train, test = split_by_percent(data, train_percentage)

  return headers, train, test

