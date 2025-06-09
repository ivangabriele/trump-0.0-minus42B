from os import path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    # Load the trained model and tokenizer from checkpoint
    checkpoint_path = path.join(path.dirname(__file__), "results/checkpoint-3")
    print(f"Loading model from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

    print("Model loaded successfully. Type 'exit' to end chat.")

    while True:
        print("═" * 80)
        input_text = input("You: ")
        if input_text.lower() == "exit":
            break

        response = generate_response(model, tokenizer, input_text)
        print("─" * 80)
        print(f"Botnald J. Trump:\n\n{response}")


if __name__ == "__main__":
    main()
