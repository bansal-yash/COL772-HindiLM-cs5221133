class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        if special_tokens == None:
            special_tokens = ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]

        self.special_tokens = special_tokens

        self.token_to_id = {}
        self.id_to_token = {}

        for i in range(len(special_tokens)):
            self.token_to_id[special_tokens[i]] = i
            self.id_to_token[i] = special_tokens[i]

        print(self.token_to_id)
        print(self.id_to_token)

    def train(self, corpus):

        return

    def encode(self, text):

        return

    def decode(self, token_ids):

        return

    def save(self, filepath):

        return

    def load(self, filepath):

        return

    def get_vocab_size(self):

        return

    def get_unk_id(self):

        return
