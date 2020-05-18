from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Sentences2Sequences(object):
    def __init__(self):
        self._tokenizer = None
        self._oov_tok = '<OOV>'
        self._max_length = 120
        self._padding_type = 'post'
        self._trunc_type = 'post'

    def preprocess(self, data):
        sentences = []
        for sentence in data:
            sentence = sentence.strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            sentence = sentence.lower()
            sentences.append(sentence)
        if self._tokenizer is None:  # during training only
            self._tokenizer = Tokenizer(oov_token=self._oov_tok)
            self._tokenizer.fit_on_texts(sentences)
        sequences = self._tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(
            sequences,
            maxlen=self._max_length,
            padding=self._padding_type,
            truncating=self._trunc_type)
        return padded
