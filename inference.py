# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the trained model and tokenizer
model_name = "mf212/mistral-7b-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Help me solve this math : 1+1 = ?."
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)
