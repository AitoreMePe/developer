```python
import config
import data_processing
import model
import database
import interaction
import utils
import csv_handler
import file_handler

def main():
    # Load configurations
    configs = config.load_configs()

    # Load data
    data_files = utils.get_data_files(configs['data_dir'])
    data = []
    for file in data_files:
        if file.endswith('.csv'):
            data.append(csv_handler.read_csv(file))
        else:
            data.append(file_handler.read_file(file))

    # Process data
    processed_data = data_processing.process_data(data)

    # Load model
    gpt_model = model.load_model(configs['model_path'])

    # Create neural database
    neural_db = database.create_database(gpt_model, processed_data)

    # Interact with the database
    while True:
        query = input("Enter your query: ")
        response = interaction.interact_with_database(neural_db, query)
        print("Response: ", response)

if __name__ == "__main__":
    main()
```