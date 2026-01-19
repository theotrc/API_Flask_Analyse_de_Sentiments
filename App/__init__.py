import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from App.utils import download_model



app = Flask(__name__)

model_path = download_model()

print("Model downloaded to:", model_path)


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

	
	print("Loading model from:", model_path)
	bert_model = tf_keras.models.load_model(model_path)

from App import views



