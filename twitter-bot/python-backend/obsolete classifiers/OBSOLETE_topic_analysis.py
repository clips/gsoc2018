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

# # Visualization
# from wordcloud import WordCloud 
# import matplotlib.pyplot as plt

def analyse()

class LDA_model():
    """ """

    def __init__(self):
        """ """
        pass

    def analyse(self, tweet):
        """ """
        pass

    @staticmethod
    def load_model(path = 'model_to_load'):
        """ """
        # Code for loading a pre-trained NMF model
        model = None # placeholder for now
        return model

class NMF_model():
    """ """

    def __init__(self):
        """ """
        pass

    def analyse(self, tweet):
        """ """
        # Firstly a tf-idf matrix is calculated
        custom_stop_words = []
        # stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)
        stop_words = stopwords.words('en').union(custom_stop_words) # NLTK may be better as scikitlearn's are weird at times
        
        vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1,1), 
                                    stop_words = stop_words)

        tfidf = vectorizer.fit_transform(preprocessed)
        # print("Created document-term matrix of size {} x {}".format(*tfidf.shape[:2]))

        nmf = decomposition.NMF(init = 'nndsvd', n_components = 3, max_iter = 200)
        W = nmf.fit_transform(tfidf)
        H = nmf.components_
        print("Generated factor W of size {} and factor H of size {}".format(W.shape, H.shape))

        feature_names = vectorizer.get_feature_names()

        n_top_words = 10

        # print top words in each topic
        for topic_idx, topic in enumerate(H):
            print("Topic #{}:".format(topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()
    
    @staticmethod
    def load_model(path = 'model_to_load'):
        """ """
        # Code for loading a pre-trained NMF model
        model = None # placeholder for now
        return model

def main():
    """ """
    # Reads a JSON config from standard input to guide execution
    stdin_config = sys.stdin.readlines()
    # Should be whole of the JSON config
    json_config = json.loads(stdin_config[0])

    # LDA or NMF
    analysis_method = json_config['analysisMethod']
    tweet = json_config['tweet']

    topic = None
    model = None
    if analysis_method == 'nmf':
        model = NMF_model.load_model()
    elif analysis_method == 'lda':
        model = LDA_model.load_model()
    else:
        raise ValueError('Unrecognized value for analysis_method: {}'.format(analysis_method))
    
    topic = model.analyse(tweet)

    # Standard stream is routed back to Node here so we can just use print()
    print(topic)
    sys.stdout.flush()

if __name__ == '__main__':
    main()    