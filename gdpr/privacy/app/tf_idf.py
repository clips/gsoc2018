import os
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
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


generate_idf_counts(
    "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy. Economy china china greatly US")
