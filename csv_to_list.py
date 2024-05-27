import csv


df_csv_file = "Dataset_uniqid_class.csv"

with open(df_csv_file, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

print(data)