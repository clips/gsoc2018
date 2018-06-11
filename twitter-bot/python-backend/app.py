import sys

import topic_analyser
import sentiment_analyser
import paraphrasing_module

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.after_request
def after_request(response):
    # Add local logging later
    pass

@app.route('/analyse/topic', methods=['POST'])
def analyse_topic():
    """ """
    json = request.get_json()
    method = json.method
    text = json.text

    topics = topic_analyser.analyse(method, text)
    
    return jsonify({'topics': topics}), 200

@app.route('/analyse/sentiment', methods=['POST'])
def analyse_sentiment():
    """ """
    json = request.get_json()
    text = json.text

    sentiments, probabilities = sentiment_analyser.analyse(text)

    return jsonify({'sentiments': sentiments, 'probabilities': probabilities}), 200

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """ """
    json = request.get_json()
    text = json.text

    paraphrased_pairs = paraphrasing_module.paraphrase(text)

    # paraphrased_pairs containt pairs of original and paraphrased phrases
    return jsonify({'paraphrased_pairs': paraphrased_pairs}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')