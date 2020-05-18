import os
import pickle
import numpy as np
import tensorflow as tf


class MyPredictor(object):
    def __init__(self, model, preprocessor):
        self._model = model
        self._preprocessor = preprocessor
        self._threshold = 0.44388688

    def predict(self, instances, **kwargs):
        inputs = np.asarray(instances)
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        outputs = self._model.predict(preprocessed_inputs).tolist()
        if kwargs.get('probabilities'):
            return outputs
        else:
            return outputs > self._threshold

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'amazon_reviews_sentiment.h5')
        model = tf.keras.models.load_model(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
