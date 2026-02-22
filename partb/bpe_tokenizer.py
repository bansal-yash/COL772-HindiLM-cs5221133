from pprint import pprint


class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        if special_tokens is None:
            special_tokens = ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]

        self.special_tokens = special_tokens

        self.token_to_id = {}
        self.id_to_token = {}

        for id, token in enumerate(special_tokens):
            self.token_to_id[token] = id
            self.id_to_token[id] = token

        self.word_freqs: dict[tuple[str], int] = {}
        self.merge_rules = []

    def cons_full_vocab(self, corpus):
        curr_id = len(self.token_to_id)

        self.token_to_id["<w>"] = curr_id
        self.id_to_token[curr_id] = "<w>"
        curr_id += 1

        for sentence in corpus:
            for char in sentence:
                if char == " ":
                    continue
                if char not in self.token_to_id:
                    self.token_to_id[char] = curr_id
                    self.id_to_token[curr_id] = char
                    curr_id += 1

    def fill_word_freqs(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            for word in words:
                word_tuple = tuple(word) + ("<w>",)

                if word_tuple in self.word_freqs:
                    self.word_freqs[word_tuple] += 1
                else:
                    self.word_freqs[word_tuple] = 1

    def train_one_iteration(self):
        pair_freqs: dict[tuple[str], int] = {}

        for word_tuple, word_freq in self.word_freqs.items():
            l = len(word_tuple)
            for i in range(l - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair in pair_freqs:
                    pair_freqs[pair] += word_freq
                else:
                    pair_freqs[pair] = word_freq

        if not pair_freqs:
            return None

        best_pair = max(pair_freqs, key=pair_freqs.get)
        self.merge_rules.append(best_pair)
        best_pair_token = best_pair[0] + best_pair[1]

        new_word_freqs: dict[tuple[str], int] = {}
        for word_tuple, word_freq in self.word_freqs.items():
            new_word_list = []
            i = 0
            l = len(word_tuple)

            while i < l:
                if i < l - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                    new_word_list.append(best_pair_token)
                    i += 2
                else:
                    new_word_list.append(word_tuple[i])
                    i += 1

            new_word_tuple = tuple(new_word_list)

            if new_word_tuple in new_word_freqs:
                new_word_freqs[new_word_tuple] += word_freq
            else:
                new_word_freqs[new_word_tuple] = word_freq

        self.word_freqs = new_word_freqs

        return best_pair_token

    def train(self, corpus):

        self.cons_full_vocab(corpus)

        self.word_freqs = {}
        self.fill_word_freqs(corpus)

        print(len(self.word_freqs))

        while len(self.token_to_id) < self.vocab_size:
            new_token = self.train_one_iteration()

            if new_token is None:
                break
            curr_num_tokens = len(self.token_to_id)
            self.token_to_id[new_token] = curr_num_tokens
            self.id_to_token[curr_num_tokens] = new_token

        print(f"Tokenizer trained till vocab size of {len(self.token_to_id)}")

        print(self.token_to_id)
        print(self.id_to_token)
        print(self.merge_rules)

    def encode(self, text):
        tokens = [self.token_to_id["<|SOS|>"]]

        for char in text:
            if char == " ":
                tokens.append(self.token_to_id["<w>"])
            elif char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.get_unk_id())

        tokens.append(self.token_to_id["<|EOS|>"])
        return tokens

    def decode(self, token_ids):
        tokens = []
        for id in token_ids:
            token = self.id_to_token[id]

            if token == "<w>":
                tokens.append(" ")
            elif token in self.special_tokens:
                continue
            else:
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
