from .outlines_adapter import create_structured_model

class StructuredGeneration : 

    def __init__(self, model, tokenizer):
        self.model = create_structured_model(model, tokenizer)

    def generate(self, prompt, schema) :

        return self.model(
            prompt,
            schema
        )
    


