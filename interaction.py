```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from config import MODEL_PATH
from database import NeuralDatabase

class Interaction:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        self.database = NeuralDatabase()

    def generate_response(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=500, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interact(self, prompt):
        response = self.generate_response(prompt)
        print(response)

    def update_database(self, data):
        self.database.update(data)

if __name__ == "__main__":
    interaction = Interaction()
    while True:
        prompt = input("Enter your query: ")
        interaction.interact(prompt)
```