
class Result(object):
    def __init__(self) -> None:
        self.transcription = None
        self.tokens_txt = None
        self.tokens_int = None
        self.ctc_posteriors = None
        self.attention_weights = None
        self.encoded_vector = None
        self.input_data = None
        self.input_data_normalized = None
    
                