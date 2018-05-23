from django.shortcuts import render
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from django.http import HttpResponse
import re
import spacy
import os
import sys
import math
from pathlib import Path
from .models import Attribute_Configuration, Attribute_Alias, Supression_Configuration, Deletion_Configuration


def preprocess(text):
    ''' Preprocesses raw text and returns a cleaned, tokenized input '''
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he shall",
        "he'll've": "he will have",
        "he's": "he has",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has",
        "I'd": "I had",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she shall",
        "she'll've": "she shall have",
        "she's": "she has",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that had",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they had",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }
    # Switched off text lower case because it gives better performance on entity recognition
    #text = text.lower()
    for contraction in contractions:
        text = re.sub(contraction, contractions[contraction], text, flags=re.I)
    ''' Splitting text into sentences and words '''
    sentences = text.split('.')
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [
        sentence for sentence in tokenized_sentences if len(sentence) > 0]
    return tokenized_sentences


def entity_recognition_stanford(tokens, base_path):
    ''' Uses the Stanford NER model wrapped in NLTK'''
    classifier_model_path = base_path + '/english.muc.7class.distsim.crf.ser.gz'
    ner_jar_path = base_path + '/stanford-ner.jar'
    stanford_tagger = StanfordNERTagger(
        classifier_model_path, ner_jar_path, encoding='utf-8')
    ner_tagged = stanford_tagger.tag(tokens)
    replaced_text_list = [word[0] if word[1] == "O" else str(
        '|'+word[1]+'|') for word in ner_tagged]
    return ' '.join(replaced_text_list)


def entity_recognition_spacy(text):
    ''' Uses the SPACY NER model. Currently written using the en_core_web_sm model '''
    spacy_model = spacy.load('en_core_web_sm')
    document = spacy_model(text)
    old_text = text
    anonymized_text = ''
    entities_in_document = document.ents
    number_of_entities = len(entities_in_document)
    ''' Function to slice and replace substrings with entity labels '''
    for index, ent in enumerate(entities_in_document):
        new_label = give_new_label(ent.label_, ent.text)
        if index is 0:
            anonymized_text += old_text[:ent.start_char] + new_label + \
                old_text[ent.end_char:entities_in_document[index+1].start_char]
        elif index is number_of_entities-1:
            anonymized_text += new_label + old_text[ent.end_char:]
        else:
            anonymized_text += new_label + \
                old_text[ent.end_char:entities_in_document[index+1].start_char]
    return anonymized_text


def give_new_label(label, text):
    ''' When given the entity label and the actual entity text, returns the replacement entity '''
    try:
        # Checking for the alias in the DB
        alias = Attribute_Alias.objects.get(alias=label)
    except Attribute_Alias.DoesNotExist:
        # return and terminate function if it does not exist
        return label
    attribute_configuration = alias.attribute
    if attribute_configuration.attribute_action == 'del':
        deletion_configuration = Deletion_Configuration.objects.get(
            attribute=attribute_configuration)
        new_label = deletion_configuration.replacement_name
        return new_label
    elif attribute_configuration.attribute_action == 'gen':
        pass
    else:
        label = give_supressed_attribute(text, attribute_configuration)
        return label


def give_supressed_attribute(text, attribute_configuration):
    supression_configuration = Supression_Configuration.objects.get(
        attribute=attribute_configuration)
    replacement_character = supression_configuration.replacement_character
    if supression_configuration.suppress_number:
        # Logic flow in case the number of bits to suppress is provided
        number = supression_configuration.suppress_number
        if number > len(text):
            # If the length of the bits to supress is greater than text length
            # then replace the string entirely with asterix's
            new_text = replacement_character*len(text)
        else:
            # otherwise shave off the last supress_number number of digits
            new_text = text[:-1*number] + replacement_character*number
        return new_text
    else:
        # Logic to follow in case the percent of bits to supress is provided
        percent = supression_configuration.suppress_percent
        number = int(math.floor(0.01*percent*len(text)))
        new_text = text[:-1*number] + replacement_character*number
        return new_text


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    base_path = str(Path(base_path).parents[0])
    text = "My name is John Oliver, I stay in India and fell sick and was admitted to Hopkins hospital."\
        " I was then hired by Google."
    '''
    #Stanford experiment
    preprocessed_text = preprocess(text)
    print(preprocessed_text)
    for sentence in preprocessed_text:
        print(entity_recognition_stanford(sentence, base_path))
    '''
    # Spacy Experiment
    preprocessed_text = preprocess(text)
    sentences = [' '.join(word) for word in preprocessed_text]
    text = '. '.join(sentences)
    print(entity_recognition_spacy(text))


def main():
    base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    base_path = str(Path(base_path).parents[0])
    text = "My name is John Oliver, I stay in India and fell sick and was admitted to Hopkins hospital."\
        " I was then hired by Google."
    '''
    #Stanford experiment
    preprocessed_text = preprocess(text)
    print(preprocessed_text)
    for sentence in preprocessed_text:
        print(entity_recognition_stanford(sentence, base_path))
    '''
    # Spacy Experiment
    preprocessed_text = preprocess(text)
    sentences = [' '.join(word) for word in preprocessed_text]
    text = '. '.join(sentences)
    print('OLD TEXT : ' + text)
    print('NEW TEXT : ' + entity_recognition_spacy(text))
