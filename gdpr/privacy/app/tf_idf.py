import os
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
from math import log
stop_words = set(stopwords.words('english'))
# set(stopwords.words('english'))


def generate_idf_counts(document_text):
    tokens = word_tokenize(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    # For the IDF counts only unique tokens required hence remove the rest
    tokens = list(set(tokens))
    idf_counts_dict = {}
    try:
        # Open the idf scores file and load the idf_counts_dict
        with open('tf_idf_scores/idf_counts.json', 'r') as fp:
            idf_counts_dict = json.load(fp)
            idf_counts_dict['number_of_documents'] += 1
    except FileNotFoundError:
        # If the file does not exist, create it
        idf_counts_dict = {'number_of_documents': 1}
        os.mkdir('tf_idf_scores')
        with open('tf_idf_scores/idf_counts.json', 'w') as fp:
            json.dump(idf_counts_dict, fp)
    # Iterate through tokens and increase their count in the dict
    for token in tokens:
        try:
            idf_counts_dict[token] += 1
        except KeyError:
            idf_counts_dict[token] = 1
    # Update the idf_counts file
    with open('tf_idf_scores/idf_counts.json', 'w') as fp:
        json.dump(idf_counts_dict, fp)


def obtain_idf_scores(document_text):
    tokens = word_tokenize(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    # For the IDF counts only unique tokens required hence remove the rest
    tokens = list(set(tokens))
    idf_score_dict = {}
    with open('tf_idf_scores/idf_counts.json', 'r') as fp:
        idf_counts_dict = json.load(fp)
    # The current IDF scoring scheme is inverse document frequency smoothened.
    # Can be expanded
    for token in tokens:
        number_of_documents = idf_counts_dict['number_of_documents']
        try:
            score = log(1 + number_of_documents /
                        (idf_counts_dict[token]))
        except KeyError:
            score = log(1 + number_of_documents)
        idf_score_dict[token] = score
    return idf_score_dict


def obtain_tf_scores(document_text):
    tokens = word_tokenize(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    size_of_document = len(tokens)
    token_counts = dict(Counter(tokens))
    token_counts['size_of_document'] = size_of_document
    return token_counts


def obtain_tf_idf_scores(document_text):
    idf_scores_dict = obtain_idf_scores(document_text)
    tf_scores_dict = obtain_tf_scores(document_text)
    size_of_document = tf_scores_dict['size_of_document']
    tf_idf_scores_dict = {}
    for word in tf_scores_dict:
        tf_scores_dict[word] = (
            float(tf_scores_dict[word]) * float(idf_scores_dict[word])) / (size_of_document)
    return tf_idf_scores_dict
