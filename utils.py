import re
import nltk
from collections import Counter

nltk.download('punkt')

def tokenize(text):
    """Tokenize a sentence into words."""
    if not isinstance(text, str):
        text = ""
    return nltk.tokenize.word_tokenize(text.lower())

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word_freq = Counter()
        self.idx = 4

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            tokens = tokenize(sentence)
            self.word_freq.update(tokens)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def numericalize(self, text):
        tokenized_text = tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokenized_text] 