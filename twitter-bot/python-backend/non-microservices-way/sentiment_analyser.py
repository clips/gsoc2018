# Helper imports
import sys, json, os
import numpy as np
import pandas as pd
import re

# Topic Analysis
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn import decomposition

# Visualization
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

# my imports
import sentiment_analysis

def main ():
    """ """
    # Reads a JSON config from standard input to guide execution
    stdin_config = sys.stdin.readlines()
    # Should be whole of the JSON config
    json_config = json.loads(stdin_config[0])

    tweet = json_config['tweet']
    sentiment = sentiment_analysis(tweet)

    # Standard stream is routed back to Node here so we can just use print()
    print(sentiment)
    sys.stdout.flush()

def sentiment_analysis():
    """ """
    

if __name__ == '__main__':
    main()    