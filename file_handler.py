```python
import os
import pandas as pd

class FileHandler:
    def __init__(self):
        self.supported_formats = {
            'csv': pd.read_csv,
            # Add more formats as needed
        }

    def read_file(self, filepath):
        _, file_extension = os.path.splitext(filepath)
        file_extension = file_extension[1:]  # Remove the dot

        if file_extension not in self.supported_formats:
            raise ValueError(f'Unsupported file format: {file_extension}')

        return self.supported_formats[file_extension](filepath)

    def write_file(self, filepath, data):
        _, file_extension = os.path.splitext(filepath)
        file_extension = file_extension[1:]  # Remove the dot

        if file_extension not in self.supported_formats:
            raise ValueError(f'Unsupported file format: {file_extension}')

        if file_extension == 'csv':
            data.to_csv(filepath, index=False)
        # Add more formats as needed
```