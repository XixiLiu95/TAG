import csv
import mmcv
from os.path import dirname
file_path = "new_generated_list.txt"
mmcv.mkdir_or_exist(dirname(file_path))
# Read the CSV file to extract the replacement column
replacement_column = []
with open('imagenet_val.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        replacement_column.append(row[0])  # Assuming the second column is indexed at 1

# Read the text file and update the second column with the replacement values
updated_lines = []
with open('imagenet2012_val_list.txt', 'r') as textfile:
    for line in textfile:
        columns = line.split()  # Split the line into columns (assuming they are separated by whitespace)
        if len(columns) >= 2:
            columns[1] = replacement_column.pop(0)  # Replace the second column
        updated_lines.append(' '.join(columns))


# Write the updated data back to the text file
with open('new_generated_list.txt', 'w') as textfile:
    for item in updated_lines:
        textfile.write(item + "\n")
print(f"The list has been saved to {file_path}")