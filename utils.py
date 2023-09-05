```python
import os
import csv
import json

def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def write_csv_file(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def check_file_exists(file_path):
    return os.path.exists(file_path)

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
```