```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPTModel:
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, temperature=0.7, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0])

if __name__ == "__main__":
    model_path = "models/GPT4All-13B-snoozy.ggmlv3.q4_0.bin"
    gpt_model = GPTModel(model_path)
    prompt = "Analyze the game data"
    print(gpt_model.generate_text(prompt))
```