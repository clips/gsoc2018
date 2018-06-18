# Helper imports
import sys, json, os
import numpy as np
import pandas as pd
import re

# Topic Analysis
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import decomposition

# Visualization
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

# my imports
from topic_analysis import LDA_model, NMF_model

def main ():
    """ """
    # Reads a JSON config from standard input to guide execution
    stdin_config = sys.stdin.readlines()
    # Should be whole of the JSON config
    json_config = json.loads(stdin_config[0])

    # LDA or NMF
    analysis_method = json_config['analysisMethod']
    tweet = json_config['tweet']

    topic = None
    if analysis_method == 'nmf':
        topic = nmf(tweet)
    elif analysis_method == 'lda':
        topic = lda(tweet)
    else:
        raise ValueError('Unrecognized value for analysis_method: {}'.format(analysis_method))
    
    # Standard stream is routed back to Node here so we can just use print()
    print(topic)
    sys.stdout.flush()

def lda(tweet):
    """ """
    lda_model = LDA_model.load_model()
    topic_prediction = lda_model.predict(tweet)

    topic = analyse_topic_prediction(topic_prediction)
    return topic

def nmf(tweet):
    """ """
    nmf_model = NMF_model.load_model()
    topic_prediction = nmf_model.predict(tweet)

    topic = analyse_topic_prediction(topic_prediction)
    return topic
    # Firstly a tf-idf matrix is calculated
    custom_stop_words = []
    # stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)
    stop_words = stopwords.words('en').union(custom_stop_words) # NLTK may be better as scikitlearn's are weird at times
    
    vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1,1), 
                                stop_words = stop_words)

    tfidf = vectorizer.fit_transform(preprocessed)
    # print("Created document-term matrix of size {} x {}".format(*tfidf.shape[:2]))


if __name__ == '__main__':
    main()    