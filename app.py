import argparse
import json

from llm.config import LLMSettings
from llm.TicketClassifier import TicketClassifier
from llm.providers.openai_compatible import OpenAICompatible


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a support ticket as structured JSON.")
    parser.add_argument("ticket", help="Support ticket text")
    args = parser.parse_args()

    settings = LLMSettings()
    classifier = TicketClassifier(OpenAICompatible(settings))
    result = classifier.classify(args.ticket)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
