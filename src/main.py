```python
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy import stats

from data_loader import load_data, match_data
from preprocessing import preprocess_data, preprocessed_data
from graph_gps import model as graph_gps_model, train as graph_gps_train, validate as graph_gps_validate, test as graph_gps_test, model_parameters as graph_gps_model_parameters, model_performance_metrics as graph_gps_model_performance_metrics
# Import other models in a similar way

def run_model(model, train, validate, test, data):
    model = train(data)
    validate(model, data)
    test(model, data)
    return model

def main():
    data = load_data('league_of_legends_match_data.csv')
    preprocessed_data = preprocess_data(data)

    graph_gps_model = run_model(graph_gps_model, graph_gps_train, graph_gps_validate, graph_gps_test, preprocessed_data)
    # Run other models in a similar way

    print("GraphGPS Model Parameters: ", graph_gps_model_parameters)
    print("GraphGPS Model Performance Metrics: ", graph_gps_model_performance_metrics)
    # Print parameters and performance metrics of other models in a similar way

if __name__ == "__main__":
    main()
```