import sys

import topic_analyser
import anger_analyser

from flask import Flask, jsonify, request

app = Flask(__name__)

# @app.after_request
# def after_request(response):
#     # Add local logging later
#     pass

@app.route('/analyse/topic', methods=['POST'])
def analyse_topic():
    """ """
    json = request.get_json()
    text = json['text']

    topics, probabilities = topic_analyser.analyse(text)
    
    return jsonify({'predictions': list(topics), 'probabilities': str(list(probabilities))})
    # return {'topics': topics, 'probabilities': probabilities}, 200

@app.route('/analyse/anger', methods=['POST'])
def analyse_sentiment():
    """ """
    json = request.get_json()
    text = json['text']

    # Mock variable setting
    # text = 'Sample text about president'

    anger = anger_analyser.analyse(text)

    return jsonify({'prediction': str(anger[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

# @app.route('/paraphrase', methods=['POST'])
# def paraphrase():
#     """ """
#     json = request.get_json()
#     text = json['text']

#     paraphrased_pairs = paraphrasing_module.paraphrase(text)

#     # paraphrased_pairs containt pairs of original and paraphrased phrases
#     return jsonify({'paraphrased_pairs': paraphrased_pairs}), 200

