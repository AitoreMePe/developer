import csv
import pandas as pd
from config import DATA_PATH

class CSVHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_csv(self):
        try:
            data = pd.read_csv(self.filepath)
            return data
        except FileNotFoundError:
            print(f"No file found at {self.filepath}")
            return None

    def write_csv(self, data, filename):
        try:
            data.to_csv(DATA_PATH + filename, index=False)
        except Exception as e:
            print(f"Error writing to CSV: {e}")

    def append_csv(self, data, filename):
        try:
            with open(DATA_PATH + filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        except Exception as e:
            print(f"Error appending to CSV: {e}")