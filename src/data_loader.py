```python
import pandas as pd

def load_data(filepath):
    match_data = pd.read_csv(filepath)
    return match_data

match_data = load_data('path_to_your_csv_file.csv')
```