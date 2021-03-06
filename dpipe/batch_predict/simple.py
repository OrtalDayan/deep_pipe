import numpy as np

from .base import BatchPredict


class Simple(BatchPredict):
    def validate(self, x, y, *, validate_fn):
        prediction, loss = validate_fn(x[None], y[None])
        return prediction[0], loss

    def predict(self, x, *, predict_fn):
        return predict_fn(x[None])[0]


class Multiclass(BatchPredict):
    def validate(self, x, y, *, validate_fn):
        prediction, loss = validate_fn(x[None], y[None])
        return np.argmax(prediction[0], axis=0), loss

    def predict(self, x, *, predict_fn):
        return np.argmax(predict_fn(x[None])[0], axis=0)
