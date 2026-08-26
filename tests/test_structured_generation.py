import os 
import sys
sys.path.append(os.path.abspath('..'))
from src.inference.model import load_model
from structured.generator import StructuredGeneration
from structured.schemas   import TicketOutput
from prompts.base   import build_ticket_classifier_prompt



def main() : 
    tokenizer, model = load_model()

    generator = StructuredGeneration(model, tokenizer)

    user_query = """Your support team was very helpful, thank you!"""

    prompt = build_ticket_classifier_prompt(user_query)

    result = generator.generate(prompt, TicketOutput)

    print(result)


if __name__ == '__main__' :
    main()