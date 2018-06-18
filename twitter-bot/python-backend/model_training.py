import os 
import re
import pickle

import pandas as pd
import numpy as np 

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.decomposition import NMF, LatentDirichletAllocation

def clean_data(data):
    data = re.sub(r"@\w+", "", data) #remove twitter handle
    data = re.sub(r"\d", "", data) # remove numbers  
    data = re.sub(r"_+", "", data) # remove consecutive underscores
    data = data.lower() # tranform to lower case    
    
    return data.strip()

def tokenize_data(data):
    tokenized_data = [" ".join(RegexpTokenizer(r'\w+'). \
                          tokenize(sentiment_data.cleaned_tweet_text[index])) \
                      for index in data.index]
    return tokenized_data

def calculate_tf_idf(data):
    stop_words = []
    stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)

    vectorizer = TfidfVectorizer(min_df = 1, ngram_range = (1,1), 
                                stop_words = stop_words)

    tf_idf = vectorizer.fit_transform(data)
    print("Created document-term matrix of size {} x {}".format(*tf_idf.shape[:2]))
    return tf_idf, vectorizer

def nmf(tf_idf, vectorizer):
    nmf = decomposition.NMF(init = 'nndsvd', n_components = 3, max_iter = 200)
    W = nmf.fit_transform(tf_idf)
    H = nmf.components_
    print("Generated factor W of size {} and factor H of size {}".format(W.shape, H.shape))

    feature_names = vectorizer.get_feature_names()
    n_top_words = 10

    for topic_idx, topic in enumerate(H):
        print("Topic #{}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words -1:-1]]))

    return nmf

def save_model(model, model_type):
    """ """
    filename = 'model/' + model_type + '_model.pkl'
    return pickle.dump(model, filename)

if __name__ == '__main__':
    columns = ['sentiment','id','date','keyword','username','tweet_text']
    dirname = os.getcwd()
    path = os.path.join(dirname, "python-backend\\datasets\\sentiment140\\sentiment140_data.csv")
    sentiment_data = pd.read_csv(path, names = columns, header = None, encoding = "ISO-8859-1")
    print(sentiment_data.columns)
    sentiment_data["cleaned_tweet_text"] = sentiment_data["tweet_text"].apply(clean_data) 
    preprocessed_data = tokenize_data(sentiment_data)
    tf_idf, vectorizer = calculate_tf_idf(preprocessed_data)
    nmf = nmf(tf_idf, vectorizer)
    
    text_to_analyse = ['Something bad has happened', 'I love that']
    tf_idf_new = vectorizer.transform(text_to_analyse)
    W_new = nmf.transform(tf_idf_new) 
