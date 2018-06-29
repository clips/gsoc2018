# Helper imports
import sys, json, os
import numpy as np
import pandas as pd
import re
import pickle

# Topic Analysis
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import decomposition

def analyse(analysis_method, text):
    """ """
    topic = None
    if analysis_method == 'nmf':
        topic = nmf(text)
    elif analysis_method == 'lda':
        topic = lda(text)
    else:
        raise ValueError('Unrecognized value for analysis_method: {}'.format(analysis_method))

    return topic
    
def lda(text):
    """ """
    # lda_model = load_model('lda')
    # topic_prediction = lda_model.predict(text)

    # mock topic prediction
    topic_prediction = re.sub("[^\w]", " ",  text).split()

    topic = analyse_topic_prediction(topic_prediction)
    return topic

def nmf(text):
    """ """
    # nmf_model = load_model('nmf')
    # topic_prediction = nmf_model.predict(text)

        # mock topic prediction
    topic_prediction = re.sub("[^\w]", " ",  text).split()

    topic = analyse_topic_prediction(topic_prediction)
    return topic


def analyse_topic_prediction(prediction):
    """ """
    if 'global' and 'warming' in prediction:
        prediction.append('global warming')
    if 'ozone' and 'layer' in prediction:
        prediction.append('ozone layer')

    for topic in prediction:
        # Naive mock method for now
        if topic in ['politics', 'war', 'state', 'politician', 'president', 'parliament', 'rights', 'laws', 'gerrymandering']:
            return 'politics'
        if topic in ['justice', 'judge', 'legal', 'illegal', 'society', 'societal', 'social', 'animal rights']:
            return 'social issues'
        if topic in ['father', 'mother', 'sister', 'brother', 'home', 'family', 'grandmother', 'grandfather']:
            return 'family'
        if topic in ['girlfriend', 'boyfriend', 'relationship', 'love', 'marriage']:
            return 'relationships'  
        if topic in ['football', 'game', 'player', 'hockey', 'tennis', 'swimming', 'running', 'cycling']:
            return 'sports'
        if topic in ['environment', 'ecology', 'global warming', 'deforestation', 'ozone layer', 'plastic ocean', 'renewable', 'anthropocene', 'extinction']:
            return 'environmental issues'
        if topic in ['hunger', 'food', 'famine', 'veganism', 'vegetarianism']:
            return 'food-related issues'  
    return 'something'

def load_model(model_type):
    """ """
    filename = 'models/' + model_type + '_model.pkl'
    return pickle.load(filename)

# # Firstly a tf-idf matrix is calculated
# custom_stop_words = []
# # stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)
# stop_words = stopwords.words('en').union(custom_stop_words) # NLTK may be better as scikitlearn's are weird at times

# vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1,1), 
#                             stop_words = stop_words)

# tfidf = vectorizer.fit_transform(preprocessed)
# # print("Created document-term matrix of size {} x {}".format(*tfidf.shape[:2]))
