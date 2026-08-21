import torch
from infrence.model import load_model
from infrence.generate import generate_text


if __name__ == "__main__" : 

    tokenizer, model = load_model()

    prompt = "what is AI"

    output = generate_text(
        tokenizer=tokenizer,
        model = model,
        prompt=prompt,
        top_p=.9,
        temperature=.2
    )

    print("\n --- generated output ---")
    print(output)

