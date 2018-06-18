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

def load_model():
    """ """
    model = None

    ###

    return model

def analyse(text):
    """ """
    sentiment, probabilities = None
    model = load_model()

    ###

    # mock analysis:
    sentiment = 'negative'
    probabilities = '0.12345'

    return sentiment, probabilities

