from flask import render_template, request, redirect, url_for, jsonify
from App import app, utils, bert_model
from logging import FileHandler, WARNING


file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
app.logger.addHandler(file_handler)

@app.route("/")
def home():
    return "api prediction de sentiment"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Accept text to predict from JSON POST {'text': ...}, form-data 'text',
    or query param ?text=...  Return JSON with predictions.
    """
    # extract text from JSON body, form or querystring
    text = None
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if data and 'text' in data:
            text = data['text']
        else:
            text = request.form.get('text') or request.values.get('text')
    else:
        text = request.args.get('text')

    if not text:
        return jsonify({'error': 'No text provided. Send JSON {"text": "..."} or form/query param "text".'}), 400

    # allow a single string or a list of strings
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, (list, tuple)):
        texts = list(text)
    else:
        return jsonify({'error': 'Invalid text format; must be string or list of strings.'}), 400

    try:
        preds = bert_model.predict(texts)

        preds_list = preds.tolist()

        for i in range(len(preds_list[0])):
                if preds_list[0][i] > 0.5:
                    preds_list[0][i] = "POSITIVE"
                else:
                    preds_list[0][i] = "NEGATIVE"
    except Exception as e:
        app.logger.exception('Prediction failed')
        return jsonify({'error': str(e)}), 500

    return jsonify({'predictions': preds_list})
