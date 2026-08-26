


class StopSequenceDetector : 

    def __init__(self, stop_token_sequence:list[list[int]]|None = None):

        self.stop_token_sequence = (stop_token_sequence or [])


    def match(
            self, 
            generated_tokens:list[int]
    ):
#          [is, ahmed]
# [my,name, is, ahmed]
        for stop_sequence in self.stop_token_sequence : 
            if not stop_sequence : 
                continue
            if len(generated_tokens) < len(stop_sequence) :
                continue
            if generated_tokens[-len(stop_sequence):] == stop_sequence :
                return len(stop_sequence)

        return 0 