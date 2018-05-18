from django.shortcuts import render
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from django.http import HttpResponse
import re
import spacy
import os
import sys
from pathlib import Path


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
    text = text.lower()
    for contraction in contractions:
        text = re.sub(contraction, contractions[contraction], text)
    ''' Splitting text into sentences and words '''
    sentences = text.split('.')
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [
        sentence for sentence in tokenized_sentences if len(sentence) > 0]
    return tokenized_sentences


def entity_recognition_stanford(tokens, base_path):
    classifier_model_path = base_path + '/english.muc.7class.distsim.crf.ser.gz'
    ner_jar_path = base_path + '/stanford-ner.jar'
    stanford_tagger = StanfordNERTagger(
        classifier_model_path, ner_jar_path, encoding='utf-8')
    ner_tagged = stanford_tagger.tag(tokens)
    replaced_text_list = [word[0] if word[1] == "O" else str(
        '|'+word[1]+'|') for word in ner_tagged]
    return ' '.join(replaced_text_list)


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    base_path = str(Path(base_path).parents[0])
    text = "My name is John Oliver, I stay in Dubai and fell sick and was admitted to Hopkins hospital."
    preprocessed_text = preprocess(text)
    print(preprocessed_text)
    for sentence in preprocessed_text:
        print(entity_recognition_stanford(sentence, base_path))
