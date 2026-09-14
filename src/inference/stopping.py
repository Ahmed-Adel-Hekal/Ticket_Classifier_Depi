class StopSequenceDetector:
    def __init__(self, stop_token_sequences: list[list[int]] | None = None):
        self.stop_token_sequences = stop_token_sequences or []

    def match(self, generated_tokens: list[int]) -> int:
        for sequence in self.stop_token_sequences:
            if sequence and len(generated_tokens) >= len(sequence):
                if generated_tokens[-len(sequence):] == sequence:
                    return len(sequence)
        return 0
