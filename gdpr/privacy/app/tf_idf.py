import os
from collections import Counter
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
from math import log
stop_words = set(stopwords.words('english'))
import spacy
# set(stopwords.words('english'))


def tokenizer_wrapper(text_to_tokenize):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text_to_tokenize)
    tokens = [token.text for token in doc]
    return tokens


def generate_idf_counts(user_id, document_text):
    ''' 
    Saves the document and wordcount to the knowledgebase
    '''
    if not os.path.exists('tf_idf_scores'):
        os.makedirs('tf_idf_scores')

    tokens = tokenizer_wrapper(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    # For the IDF counts only unique tokens required hence remove the rest
    tokens = list(set(tokens))
    idf_counts_dict = {}
    file_path = 'tf_idf_scores/' + str(user_id) + '_idf_counts.json'
    try:
        # Open the idf scores file and load the idf_counts_dict
        with open(file_path, 'r') as fp:
            idf_counts_dict = json.load(fp)
            idf_counts_dict['number_of_documents'] += 1
    except FileNotFoundError:
        # If the file does not exist, create it
        idf_counts_dict = {'number_of_documents': 1}
        with open(file_path, 'w') as fp:
            json.dump(idf_counts_dict, fp)
    # Iterate through tokens and increase their count in the dict
    for token in tokens:
        try:
            idf_counts_dict[token] += 1
        except KeyError:
            idf_counts_dict[token] = 1
    # Update the idf_counts file
    with open(file_path, 'w') as fp:
        json.dump(idf_counts_dict, fp)


def obtain_idf_scores(user_id, document_text):
    tokens = tokenizer_wrapper(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    # For the IDF counts only unique tokens required hence remove the rest
    tokens = list(set(tokens))
    idf_score_dict = {}
    file_path = 'tf_idf_scores/' + str(user_id) + '_idf_counts.json'
    with open(file_path, 'r') as fp:
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
    tokens = tokenizer_wrapper(document_text)
    regex_pattern_for_tokens = re.compile('[a-zA-Z]')
    # List comprehension below removes non alphabet characters and stopwords
    tokens = [token.lower() for token in tokens if token.lower()
              not in stop_words and re.match(regex_pattern_for_tokens, token)]
    size_of_document = len(tokens)
    token_counts = dict(Counter(tokens))
    return (size_of_document, token_counts)


def obtain_tf_idf_scores(user_id, document_text):
    '''
    Obtains the tf_idf scores tokenwise as a dictionary
    '''
    idf_scores_dict = obtain_idf_scores(user_id, document_text)
    tf_scores_result = obtain_tf_scores(document_text)
    tf_scores_dict = tf_scores_result[1]
    size_of_document = tf_scores_result[0]
    tf_idf_scores_dict = {}
    for word in tf_scores_dict:
        tf_idf_scores_dict[word] = (
            float(tf_scores_dict[word]) * float(idf_scores_dict[word])) / (size_of_document)
    return tf_idf_scores_dict
