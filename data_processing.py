```python
import pandas as pd
from csv_handler import CSVHandler
from file_handler import FileHandler

class DataProcessing:
    def __init__(self):
        self.csv_handler = CSVHandler()
        self.file_handler = FileHandler()

    def load_data(self, filepath):
        if filepath.endswith('.csv'):
            data = self.csv_handler.read_csv(filepath)
        else:
            data = self.file_handler.read_file(filepath)
        return data

    def preprocess_data(self, data):
        # Remove any rows with missing data
        data = data.dropna()

        # Convert categorical data to numerical data
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].astype('category').cat.codes

        return data

    def save_processed_data(self, data, filepath):
        if filepath.endswith('.csv'):
            self.csv_handler.write_csv(data, filepath)
        else:
            self.file_handler.write_file(data, filepath)
```
