import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager


app = Flask(__name__)

# During testing we avoid importing heavy TensorFlow packages and loading
# the model at import time. CI/tests should set `TESTING=1` to use a stub.
if os.environ.get('TESTING') == '1':
	class _DummyModel:
		def predict(self, texts):
			import numpy as _np
			return _np.array([[0.9, 0.1]])

	bert_model = _DummyModel()
else:
	import tensorflow as tf
	import tensorflow_text
	import tf_keras

	bert_model = tf_keras.models.load_model(os.path.join('App', 'models', 'saved_model2'))

from App import views



