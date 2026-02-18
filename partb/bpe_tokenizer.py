from pprint import pprint


class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        if special_tokens == None:
            special_tokens = ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]

        self.special_tokens = special_tokens

        self.token_to_id = {}
        self.id_to_token = {}

        for id, token in enumerate(special_tokens):
            self.token_to_id[token] = id
            self.id_to_token[id] = token

    def cons_full_vocab(self, corpus):
        curr_id = len(self.token_to_id)

        for sentence in corpus:
            for char in sentence:
                if char not in self.token_to_id:
                    self.token_to_id[char] = curr_id
                    self.id_to_token[curr_id] = char
                    curr_id += 1

    def train(self, corpus):

        # Construct the full vocabulary from the corpus
        self.cons_full_vocab(corpus)

        pprint(self.token_to_id)
        pprint(self.id_to_token)

    def encode(self, text):
        tokens = [self.token_to_id["<|SOS|>"]]

        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.get_unk_id())

        tokens.append(self.token_to_id["<|EOS|>"])
        return tokens

    def decode(self, token_ids):
        tokens = []
        for id in token_ids:
            token = self.id_to_token[id]

            if token not in self.special_tokens:
                tokens.append(token)

        return "".join(tokens)

    def save(self, filepath):

        return

    def load(self, filepath):

        return

    def get_vocab_size(self):
        return len(self.token_to_id)

    def get_unk_id(self):
        return self.token_to_id["<|UNK|>"]
