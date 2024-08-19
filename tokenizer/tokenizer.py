
class Tokenizer():

    def __init__(self, corpus):
        super().__init__()
        self._vocab_size = 0
        self.corpus = corpus
        self._stoi = None
        self._itos = None

    def get_vocab_size(self):
        return self._vocab_size


    def train(self):
        chars = sorted(list(set(self.corpus)))
        self._vocab_size = len(chars)
        self._stoi = { ch:i for i,ch in enumerate(chars) }
        self._itos = { i:ch for i,ch in enumerate(chars) }

    def encode(self, text):
         # encoder: take a string, output a list of integers
        return [self._stoi[c] for c in text]
    
    def decode(self, tokenIds):
        # decoder: take a list of integers, output a string
        return ''.join([self._itos[i] for i in tokenIds])
