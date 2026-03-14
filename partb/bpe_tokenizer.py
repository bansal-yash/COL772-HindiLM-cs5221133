import os
import json
import string


class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        self.vocab_size = vocab_size
        main_special_tokens = ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]

        if special_tokens is None:
            self.special_tokens = main_special_tokens
        else:
            extra = [t for t in special_tokens if t not in main_special_tokens]
            self.special_tokens = main_special_tokens + extra

        self.eow_token = "<|w|>"
        self.token_to_id = {}
        self.id_to_token = {}

        for id, token in enumerate(self.special_tokens + [self.eow_token]):
            self.token_to_id[token] = id
            self.id_to_token[id] = token

        self.merge_rules = []

    def cons_full_vocab(self, corpus):
        curr_id = len(self.token_to_id)
        chars = set(string.printable) - {" ", "\x0b", "\x0c"}

        for cp in range(0x0900, 0x0980):
            chars.add(chr(cp))

        for sentence in corpus:
            for char in sentence:
                if char != " ":
                    chars.add(char)

        for char in sorted(chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = curr_id
                self.id_to_token[curr_id] = char
                curr_id += 1

    def fill_word_freqs(self, corpus, word_freqs):
        for sentence in corpus:
            words = sentence.split(" ")
            for word in words:
                word_tuple = tuple(word) + (self.eow_token,)

                if word_tuple in word_freqs:
                    word_freqs[word_tuple] += 1
                else:
                    word_freqs[word_tuple] = 1

    def train_one_iteration(self, word_freqs):
        pair_freqs: dict[tuple[str], int] = {}

        for word_tuple, word_freq in word_freqs.items():
            l = len(word_tuple)
            for i in range(l - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                if pair in pair_freqs:
                    pair_freqs[pair] += word_freq
                else:
                    pair_freqs[pair] = word_freq

        if not pair_freqs:
            return None, word_freqs

        best_pair = max(pair_freqs, key=pair_freqs.get)
        self.merge_rules.append(best_pair)
        best_pair_token = best_pair[0] + best_pair[1]

        new_word_freqs: dict[tuple[str], int] = {}
        for word_tuple, word_freq in word_freqs.items():
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

        return best_pair_token, new_word_freqs

    def train(self, corpus):
        self.cons_full_vocab(corpus)

        word_freqs: dict[tuple[str], int] = {}
        self.fill_word_freqs(corpus, word_freqs)

        while len(self.token_to_id) < self.vocab_size:
            new_token, word_freqs = self.train_one_iteration(word_freqs)

            if new_token is None:
                break
            curr_num_tokens = len(self.token_to_id)
            self.token_to_id[new_token] = curr_num_tokens
            self.id_to_token[curr_num_tokens] = new_token

        print(f"Tokenizer trained till vocab size of {len(self.token_to_id)}")

    def encode(self, text):
        tokens = [self.token_to_id["<|SOS|>"]]

        words = text.split(" ")
        for word in words:
            word_tuple = tuple(word) + (self.eow_token,)
            for rule in self.merge_rules:
                new_word_list = []
                i = 0
                l = len(word_tuple)

                while i < l:
                    if i < l - 1 and (word_tuple[i], word_tuple[i + 1]) == rule:
                        new_word_list.append(rule[0] + rule[1])
                        i += 2
                    else:
                        new_word_list.append(word_tuple[i])
                        i += 1

                word_tuple = tuple(new_word_list)

            for token in word_tuple:
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    tokens.append(self.get_unk_id())

        tokens.append(self.token_to_id["<|EOS|>"])
        return tokens

    def decode(self, token_ids):
        tokens = []
        for id in token_ids:
            token = self.id_to_token[id]

            if token in self.special_tokens:
                continue
            else:
                tokens.append(token)

        sentence = "".join(tokens)
        sentence = sentence.replace(self.eow_token, " ")

        return sentence[:-1]

    def save(self, filepath):
        os.makedirs(filepath, exist_ok=True)
        save_path = os.path.join(filepath, "bpe_tokenizer.json")

        save_data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "eow_token": self.eow_token,
            "token_to_id": self.token_to_id,
            "merge_rules": [list(pair) for pair in self.merge_rules],
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)

    def load(self, filepath):
        save_path = os.path.join(filepath, "bpe_tokenizer.json")

        if not os.path.exists(save_path):
            print(f"Tokenizer file not found at: {save_path}")
            return False

        with open(save_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        self.special_tokens = saved_data["special_tokens"]
        self.eow_token = saved_data["eow_token"]
        self.token_to_id = saved_data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.merge_rules = [tuple(pair) for pair in saved_data["merge_rules"]]

        return True

    def get_vocab_size(self):
        return len(self.token_to_id)

    def get_unk_id(self):
        return self.token_to_id["<|UNK|>"]
