```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(match_data):
    # Drop unnecessary columns
    match_data = match_data.drop(columns=['Column1', 'Column2'])  # replace 'Column1', 'Column2' with actual column names to be dropped

    # Handle missing values
    match_data = match_data.fillna(method='ffill')

    # Convert categorical data to numerical data
    match_data = pd.get_dummies(match_data)

    # Normalize numerical data
    scaler = StandardScaler()
    match_data = pd.DataFrame(scaler.fit_transform(match_data), columns = match_data.columns)

    return match_data

# Load the data
from data_loader import load_data
match_data = load_data('path_to_csv_file')  # replace 'path_to_csv_file' with actual path to the csv file

# Preprocess the data
preprocessed_data = preprocess_data(match_data)

# Export the preprocessed data
preprocessed_data.to_csv('preprocessed_data.csv', index=False)
```