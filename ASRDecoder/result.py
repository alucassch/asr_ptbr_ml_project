
class Result(object):
    def __init__(self) -> None:
        self.text = None
        self.tokens_txt = None
        self.tokens_int = None
        self.ctc_posteriors = None
        self.attention_weights = None
        self.encoded_vector = None
        self.audio_samples = None
        self.mel_features = None
    
                