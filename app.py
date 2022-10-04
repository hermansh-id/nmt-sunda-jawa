import tensorflow_text as tf_text
import numpy as np
import tensorflow as tf
from flask import Flask, Response, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_id_ja = tf.saved_model.load('../model/id_ja')
model_id_su = tf.saved_model.load('../model/id_su')
model_ja_id = tf.saved_model.load('../model/ja_id')
model_su_id = tf.saved_model.load('../model/su_id')
model_ja_su = tf.saved_model.load('../model/ja_su')
model_su_ja = tf.saved_model.load('../model/su_ja')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def predict_word(word, idmodel):
    text = tf.constant([word])
    if idmodel == 'id_ja':
      result = model_id_ja.tf_translate(text)
    elif idmodel == 'id_su':
      result = model_id_su.tf_translate(text)
    elif idmodel == 'ja_id':
      result = model_ja_id.tf_translate(text)
    elif idmodel == 'su_id':
      result = model_su_id.tf_translate(text)
    elif idmodel == 'ja_su':
      result = model_ja_su.tf_translate(text)
    elif idmodel == 'su_ja':
      result = model_su_ja.tf_translate(text)
    else:
      return 'error'
    return result['text'][0].numpy().decode()

@app.route('/', methods=['POST'])
@cross_origin()
def predict():
  try:
    data = request.json
    word = data['word']
    idmodel = data['idmodel']
    result = predict_word(word, idmodel)
    if result != 'error':
      return jsonify(result=result, original=word, model=idmodel), 200
    else:
      return Response("{'result': 'languange not found'}", status=404, mimetype='application/json')
  except:
    return Response("{'result': 'error'}", status=500, mimetype='application/json')


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
