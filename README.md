# LLM Ticket Platform

A small provider-agnostic LLM application for classifying customer-support tickets. The repository also contains an independent custom text-generation module for studying autoregressive inference and decoding.

## Architecture

```text
Application / TicketClassifier
        |
        v
Prompt builders -> GenerationRequest -> LLMGateway
                                      |
                         +------------+------------+
                         |                         |
                 OpenAICompatible           Local/custom backend
                         |                         |
                         +------------+------------+
                                      v
                              Raw model text
                                      |
                                      v
                          JSON parsing + Pydantic
                                      |
                                      v
                               TicketOutput
```

The remote-provider application path and the custom local inference engine are intentionally separate. The custom decoder is not silently used by the OpenAI-compatible provider or by the optional Outlines adapter.

## Setup

1. Create an environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

2. Install the core dependencies:

   ```bash
   pip install -e ".[test]"
   ```

3. Copy `.env.example` to `.env` and set `LLM_API_KEY` and `LLM_MODEL` if using a remote provider.

4. Run tests:

   ```bash
   pytest -q
   ```

## Example: remote provider

```python
from llm import LLMGateway, TicketClassifier
from llm.config import LLMSettings
from llm.providers import OpenAICompatible

settings = LLMSettings()
provider = OpenAICompatible(settings)
gateway = LLMGateway(
    provider,
    max_attempts=settings.max_attempts,
    base_delay=settings.base_delay,
)
classifier = TicketClassifier(gateway)

result = classifier.classify("I was charged twice for the same order.")
print(result.model_dump_json(indent=2))
```

## Optional local inference

Install the optional dependencies:

```bash
pip install -e ".[local]"
```

Then use `src.inference.generate_text` with a compatible tokenizer and causal language model. The decoder supports temperature, Top-K, Top-P, logit bias, EOS handling, and stop sequences.

## Security

Never commit `.env` or real provider credentials. Use `.env.example` as the configuration template.

## Run from the command line

Copy `.env.example` to `.env`, configure the provider values, install the package, then run:

```bash
python app.py "I was charged twice for the same order."
```

## Evaluation

The evaluation module exposes `evaluate_classifier(rows, classify)` and calculates valid-output rate, accuracy, and macro-F1 for the predicted category.

## Local engine integration

`llm.providers.local.LocalLLM` adapts the custom generation engine to the common `LLM` interface. Local inference dependencies are optional and must be installed separately.
