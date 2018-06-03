from django.shortcuts import render
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from django.http import HttpResponse, HttpResponseRedirect
import re
import spacy
import os
import sys
import math
from pathlib import Path
from .models import Attribute_Configuration, Attribute_Alias, Supression_Configuration, Deletion_Configuration, Regex_Pattern
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout


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
    # text = text.lower()
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
        '|' + word[1] + '|') for word in ner_tagged]
    return ' '.join(replaced_text_list)


def entity_recognition_spacy(text, user):
    ''' Uses the SPACY NER model. Currently written using the en_core_web_sm model '''
    spacy_model = spacy.load('en_core_web_sm')
    document = spacy_model(text)
    old_text = text
    anonymized_text = ''
    entities_in_document = document.ents
    number_of_entities = len(entities_in_document)
    ''' Function to slice and replace substrings with entity labels '''
    for index, ent in enumerate(entities_in_document):
        new_label = give_new_label(ent.label_, ent.text, user)
        if index is 0:
            anonymized_text += old_text[:ent.start_char] + new_label + \
                old_text[ent.end_char:entities_in_document[
                    index + 1].start_char]
        elif index is number_of_entities - 1:
            anonymized_text += new_label + old_text[ent.end_char:]
        else:
            anonymized_text += new_label + \
                old_text[ent.end_char:entities_in_document[
                    index + 1].start_char]
    return anonymized_text


def give_new_label(label, text, user):
    ''' When given the entity label and the actual entity text, returns the replacement entity '''
    try:
        # Checking for the alias in the DB
        alias = Attribute_Alias.objects.get(alias=label, user=user)
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
            new_text = replacement_character * len(text)
        else:
            # otherwise shave off the last supress_number number of digits
            new_text = text[:-1 * number] + replacement_character * number
        return new_text
    else:
        # Logic to follow in case the percent of bits to supress is provided
        percent = supression_configuration.suppress_percent
        number = int(math.floor(0.01 * percent * len(text)))
        new_text = text[:-1 * number] + replacement_character * number
        return new_text


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    base_path = str(Path(base_path).parents[0])
    text = "My name is John Oliver, I stay in India and fell sick and was admitted to Hopkins hospital."\
        " I was then hired by Google."
    '''
    # Stanford experiment
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
    # Stanford experiment
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


def register_user(request):
    if request.method == 'POST':
        username = request.POST.get('email')
        first_name = request.POST.get('fname')
        last_name = request.POST.get('lname')
        password = request.POST.get('password')
        user = User.objects.create_user(
            username=username, first_name=first_name, last_name=last_name)
        user.set_password(password)
        user.save()
        login(request, user)
        return HttpResponseRedirect('/dashboard')
    else:
        return render(request, 'register.html')


def login_user(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('/dashboard')
    else:
        if request.method == 'POST':
            username = request.POST.get('email')
            password = request.POST.get('password')
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                return HttpResponseRedirect('/dashboard')
        else:
            return render(request, 'login.html')


def add_attribute(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            user = request.user
            attribute_title = request.POST.get('title')
            attribute_action = request.POST.get('attribute_action')
            attribute = Attribute_Configuration.objects.create(
                attribute_title=attribute_title, attribute_action=attribute_action, user=user)
            attribute.clean()
            attribute.save()
            if attribute_action == 'supp':
                return HttpResponseRedirect('/add_suppression_configuration/' + str(attribute.id) + '/')
            elif attribute_action == 'gen':
                return HttpResponseRedirect('/add_generalization_configuration/' + str(attribute.id) + '/')
            elif attribute_action == 'del':
                return HttpResponseRedirect('/add_deletion_configuration/' + str(attribute.id) + '/')
            else:
                return HttpResponse('ILLEGAL')
            return HttpResponseRedirect('/dashboard')
        else:
            return render(request, 'add_attribute.html')
    else:
        return HttpResponseRedirect('/login')


def add_suppression_configuration(request, id):
    if request.user.is_authenticated:
        user = request.user
        attribute_configuration = Attribute_Configuration.objects.filter(
            id=id, user=user, attribute_action='supp')
        print('PASSSSSS pehle')
        if len(attribute_configuration) > 0:
            print('PASSSSSS')
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                supression_configuration, exists = Supression_Configuration.objects.get_or_create(
                    attribute=attribute)
                if request.POST.get('suppress_number'):
                    #suppress_number = int(request.POST.get('suppress_number').strip())
                    supression_configuration.suppress_number = int(request.POST.get(
                        'suppress_number'))
                    supression_configuration.suppress_percent = None
                if request.POST.get('suppress_percent'):
                    #suppress_percent = int(request.POST.get('suppress_percent').strip())
                    supression_configuration.suppress_percent = int(request.POST.get(
                        'suppress_percent'))
                    supression_configuration.suppress_number = None
                if request.POST.get('replacement_character'):
                    supression_configuration.replacement_character = request.POST.get(
                        'replacement_character')
                supression_configuration.clean()
                supression_configuration.save()
                return HttpResponseRedirect('/dashboard')
            else:
                return render(request, 'add_supression_configuration.html', {'attribute': attribute})
        else:
            return HttpResponseRedirect('/dashboard')
    else:
        return HttpResponseRedirect('/login')


def add_deletion_configuration(request, id):
    if request.user.is_authenticated:
        user = request.user
        attribute_configuration = Attribute_Configuration.objects.filter(
            id=id, user=user, attribute_action='del')
        if len(attribute_configuration) > 0:
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                deletion_configuration, exists = Deletion_Configuration.objects.get_or_create(
                    attribute=attribute)
                deletion_configuration.replacement_name = request.POST.get(
                    'replacement_name')
                deletion_configuration.save()
                return HttpResponseRedirect('/dashboard')
            else:
                return render(request, 'add_deletion_configuration.html', {'attribute': attribute})
        else:
            return HttpResponseRedirect('/dashboard')
    else:
        return HttpResponseRedirect('/login')


def add_alias(request, id):
    if request.user.is_authenticated:
        user = request.user
        attribute_configuration = Attribute_Configuration.objects.filter(
            id=id, user=user)
        if len(attribute_configuration) > 0:
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                alias = request.POST.get('alias')
                attribute_alias = Attribute_Alias.objects.create(
                    alias=alias, attribute=attribute, user=user)
                attribute_alias.save()
                return HttpResponseRedirect('/dashboard')
            else:
                return render(request, 'add_alias.html', {'attribute': attribute})
        else:
            return HttpResponseRedirect('/dashboard')
    else:
        return HttpResponseRedirect('/login')


def show_dashboard(request):
    if request.user.is_authenticated:
        user = request.user
        attributes = Attribute_Configuration.objects.filter(user=user)
        aliases = Attribute_Alias.objects.filter(user=user)
        for attribute in attributes:
            if attribute.attribute_action == 'supp':
                attribute.link = '/add_suppression_configuration/' + \
                    str(attribute.id) + '/'
            if attribute.attribute_action == 'del':
                attribute.link = '/add_deletion_configuration/' + \
                    str(attribute.id) + '/'
            if attribute.attribute_action == 'gen':
                attribute.link = '/add_generalization_configuration/' + \
                    str(attribute.id) + '/'

        return render(request, 'dashboard.html', {'user': user, 'attributes': attributes, 'aliases': aliases})
    else:
        return HttpResponseRedirect('/login')


def regex_based_anonymization(user, text):
    patterns = Regex_Pattern.objects.filter(user=user)
    new_text = text
    for pattern in patterns:
        regular_expression = re.compile(pattern.regular_expression)
        list_of_matches = re.findall(regular_expression, text)
        if list_of_matches:
            attribute_configuration = pattern.attribute
            for match in list_of_matches:
                if attribute_configuration.attribute_action == 'del':
                    # The lookup can be shifted outside the loop and optimized
                    # To be done later
                    deletion_configuration = Deletion_Configuration.objects.get(
                        attribute=attribute_configuration)
                    replacement_text = deletion_configuration.replacement_name
                elif attribute_configuration.attribute_action == 'supp':
                    replacement_text = give_supressed_attribute(
                        match, attribute_configuration)
                else:
                    replacement_text = match
                new_text = new_text.replace(match, replacement_text)
    return new_text


def anonymize(request):
    if request.user.is_authenticated:
        user = request.user
        if request.method == 'POST':
            text_to_anonymize = request.POST.get('text_to_anonymize')
            anonymized_text = regex_based_anonymization(
                user, text_to_anonymize)
            anonymized_text = entity_recognition_spacy(anonymized_text, user)
            return render(request, 'anonymize.html', {'anonymized_text': anonymized_text, 'show_output': True, 'text_to_anonymize': text_to_anonymize})
        else:
            return render(request, 'anonymize.html')

    else:
        return HttpResponseRedirect('/login')


def logout_user(request):
    if request.user.is_authenticated:
        logout(request)
        return HttpResponseRedirect('/login')
    return HttpResponseRedirect('/login')


def add_regex_pattern(request, id):
    if request.user.is_authenticated:
        user = request.user
        attribute_configuration = Attribute_Configuration.objects.filter(
            id=id, user=user)
        if len(attribute_configuration) > 0:
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                regular_expression = request.POST.get('regular_expression')
                regex_pattern = Regex_Pattern.objects.create(
                    regular_expression=regular_expression, attribute=attribute, user=user)
                regex_pattern.save()
                return HttpResponseRedirect('/dashboard')
            else:
                return render(request, 'add_regex_pattern.html', {'attribute': attribute})
        else:
            return HttpResponseRedirect('/dashboard')
    else:
        return HttpResponseRedirect('/login')
