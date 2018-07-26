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
from .models import Attribute_Configuration, Attribute_Alias, \
    Supression_Configuration, Deletion_Configuration, \
    Regex_Pattern, Generalization_Configuration, TF_IDF_configuration
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from nltk.corpus import wordnet as wn
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view
from re import sub
from django.http import JsonResponse
from pymagnitude import Magnitude
from . import tf_idf
from json import dumps
from nltk.corpus import stopwords
from django.core.files.storage import FileSystemStorage
from django.conf import settings
stop_words = set(stopwords.words('english'))


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
    if number_of_entities is 1:
        ent = entities_in_document[0]
        new_label = give_new_label(ent.label_, ent.text, user)
        anonymized_text = old_text[:ent.start_char] + \
            new_label + old_text[ent.end_char:]
    else:
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
    new_label_dict = {}
    new_label = ''
    try:
        # Checking for the alias in the DB
        alias = Attribute_Alias.objects.get(alias=label, user=user)
    except Attribute_Alias.DoesNotExist:
        # return and terminate function if it does not exist
        new_label_dict['has_new_label'] = False
        new_label_dict['new_label'] = text
        return new_label_dict
    attribute_configuration = alias.attribute
    if attribute_configuration.attribute_action == 'del':
        deletion_configuration = Deletion_Configuration.objects.get(
            attribute=attribute_configuration)
        new_label = deletion_configuration.replacement_name
    elif attribute_configuration.attribute_action == 'gen':
        new_label = give_generalized_attribute(
            attribute_configuration, user, text)
    else:
        new_label = give_supressed_attribute(text, attribute_configuration)
    # If an alias for the attribute is found, add it to the new_label_dict and
    # return
    new_label_dict['new_label'] = new_label
    new_label_dict['has_new_label'] = True
    return new_label_dict


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
        if len(attribute_configuration) > 0:
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                supression_configuration, exists = Supression_Configuration.objects.get_or_create(
                    attribute=attribute)
                if request.POST.get('suppress_number'):
                    # suppress_number =
                    # int(request.POST.get('suppress_number').strip())
                    supression_configuration.suppress_number = int(request.POST.get(
                        'suppress_number'))
                    supression_configuration.suppress_percent = None
                if request.POST.get('suppress_percent'):
                    # suppress_percent =
                    # int(request.POST.get('suppress_percent').strip())
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
        regex_patterns = Regex_Pattern.objects.filter(user=user)
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

        return render(request, 'dashboard.html', {'user': user, 'attributes': attributes, 'aliases': aliases, 'regex_patterns': regex_patterns})
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


def extract_wordvec_generalization(word, path_to_word_vectors, neighbor_number):
    vectors = Magnitude(path_to_word_vectors)
    generalized_attribute = vectors.most_similar(word, topn=neighbor_number)[
        neighbor_number - 1][0]
    return generalized_attribute


def extract_part_holonym(word, escalation_level):
    if escalation_level == 1:
        if isinstance(word, str):
            synset_word = wn.synsets(word)[0]
        else:
            synset_word = word
        holonyms = synset_word.part_holonyms()
        try:
            return holonyms[0].lemmas()[0].name()
        except (ValueError, IndexError):
            return word
    else:
        if isinstance(word, str):
            synset_word = wn.synsets(word)[0]
        else:
            synset_word = word
        holonym = synset_word.part_holonyms()[0]
        return extract_part_holonym(holonym, escalation_level - 1)


def give_generalized_attribute(attribute_configuration, user, text):
    escalation_level = 4
    path_to_word_vectors = "/home/rudresh/Documents/gsoc2018/glove.6B.100d.magnitude"
    neighbor_number = 2
    generalization_configuration = Generalization_Configuration.objects.get(
        attribute=attribute_configuration)
    if generalization_configuration.generalization_action == 'wordvec':
        generalized_attribute = extract_wordvec_generalization(
            text, path_to_word_vectors, neighbor_number)
        return generalized_attribute
    else:
        generalized_attribute = extract_part_holonym(text, escalation_level)
        return generalized_attribute


def add_generalization_configuration(request, id):
    if request.user.is_authenticated:
        user = request.user
        attribute_configuration = Attribute_Configuration.objects.filter(
            id=id, user=user, attribute_action='gen')
        if len(attribute_configuration) > 0:
            attribute = attribute_configuration[0]
            if request.method == 'POST':
                generalization_configuration, exists = Generalization_Configuration.objects.get_or_create(
                    attribute=attribute)
                generalization_configuration.generalization_action = request.POST.get(
                    'generalization_action')
                generalization_configuration.clean()
                generalization_configuration.save()
                return HttpResponseRedirect('/dashboard')
            else:
                return render(request, 'add_generalization_configuration.html', {'attribute': attribute})
        else:
            return HttpResponseRedirect('/dashboard')
    else:
        return HttpResponseRedirect('/login')


def api_token_management(request):
    if request.user.is_authenticated:
        user = request.user
        tokens = Token.objects.filter(user=user)
        return render(request, 'api_token_management.html', {'user': user, 'tokens': tokens})
    else:
        return HttpResponseRedirect('/login')


@api_view(['POST', 'GET'])
def anonymize_text_api(request):
    header_token = request.META.get('HTTP_AUTHORIZATION', None)
    if header_token is not None:
        try:
            token = sub('Token ', '', request.META.get(
                'HTTP_AUTHORIZATION', None))
            token_obj = Token.objects.get(key=token)
            request.user = token_obj.user
            user = request.user
        except Token.DoesNotExist:
            return HttpResponse('INVALID TOKEN')

        text_to_anonymize = request.POST.get('text_to_anonymize')
        anonymized_text = regex_based_anonymization(user, text_to_anonymize)
        anonymized_text = entity_recognition_spacy(anonymized_text, user)
        response_dict = {'anonymized_text': anonymized_text}
        return JsonResponse(response_dict)
    else:
        return HttpResponse("ERROR, ADD A HEADER TOKEN")


def regenerate_api_token(request):
    if request.user.is_authenticated:
        user = request.user
        instances = Token.objects.filter(user=user)
        if len(instances) > 0:
            instances[0].delete()
            Token.objects.create(user=user)
            return HttpResponseRedirect('/api_token_management')
        else:
            return HttpResponse('INVALID')
    else:
        return HttpResponseRedirect('/login')


def token_level_anon(text_to_anonymize, user):
    spacy_model = spacy.load('en_core_web_sm')
    document = spacy_model(text_to_anonymize)
    token_response = []
    index = 0
    while(index < len(document)):
        word = document[index]
        word_text = word.text
        if word.ent_type_:
            if word.ent_iob_ == 'B':
                # placeholder for the replacement function
                entity_type = word.ent_type_
                replacement_dict = give_new_label(entity_type, word_text, user)
                replacement = replacement_dict['new_label']
                is_replaced = replacement_dict['has_new_label']
                token_dict = {'token': word_text, 'is_entity': True,
                              'entity_type': entity_type, 'replacement': replacement, 'is_replaced': is_replaced}
                token_response.append(token_dict)
            else:
                # handling multi token entities,I.E when the entity type is
                # "I"
                temporary_index = index
                '''
                Acessing the first token in the entity, that is current last
                entry in array
                '''
                word_text = token_response[-1]['token']
                while(temporary_index < len(document) and document[temporary_index].ent_iob_ == 'I'):
                    # appending all the words that are a part of that
                    # entity
                    word_text = word_text + ' ' + \
                        document[temporary_index].text
                    temporary_index += 1
                # placeholder for the replacement function
                entity_type = word.ent_type_
                replacement_dict = give_new_label(entity_type, word_text, user)
                replacement = replacement_dict['new_label']
                is_replaced = replacement_dict['has_new_label']
                token_dict = {'token': word_text, 'is_entity': True,
                              'entity_type': entity_type, 'replacement': replacement, 'is_replaced': is_replaced}
                # Deleting the beginning token entry and replacing with
                # entire string
                del token_response[-1]
                token_response.append(token_dict)
                # Skipping all the  I entities covered
                index = temporary_index

        else:
            replacement = ''
            token_dict = {'token': word_text, 'is_entity': False,
                          'entity_type': word.ent_type_, 'replacement': replacement, 'is_replaced': False}
            token_response.append(token_dict)
        index += 1
    response = {'response': token_response, 'original_text': text_to_anonymize}
    return response


@api_view(['POST'])
def token_level_api(request):
    header_token = request.META.get('HTTP_AUTHORIZATION', None)
    if header_token is not None:
        try:
            token = sub('Token ', '', request.META.get(
                'HTTP_AUTHORIZATION', None))
            token_obj = Token.objects.get(key=token)
            request.user = token_obj.user
            user = request.user
        except Token.DoesNotExist:
            return HttpResponse('INVALID TOKEN')
        text_to_anonymize = request.POST.get('text_to_anonymize')
        # This is NER based detection and anonymization
        response = token_level_anon(text_to_anonymize, user)
        # If the user has set the flag for TF-IDF anonymization, only then will
        # it take place
        if 'tfidf_anonymize' in request.POST:
            if request.POST.get('tfidf_anonymize') == 'True' or request.POST.get('tfidf_anonymize') == 'true':
                # Passing to TF-IDF BASED RARE TOKEN DETECTION
                response = token_level_tf_idf_anonymize(response, user)
        return JsonResponse(response)


def add_document_to_knowledgebase(request):
    ''' Wrapper to add documents to the TF-IDF knowledgebase '''
    if request.user.is_authenticated:
        user = request.user
        user_id = user.id
        if request.method == 'POST':
            document_text = request.POST.get('document_text')
            tf_idf.generate_idf_counts(user_id, document_text)
            return render(request, 'add_document_to_knowledgebase.html', {'success': True})
        else:
            return render(request, 'add_document_to_knowledgebase.html')
    else:
        return HttpResponseRedirect('/login')


def tf_idf_anonymize(request):
    ''' Trial function with seperate template. Template to be merged later '''
    if request.user.is_authenticated:
        user = request.user
        if request.method == 'POST':
            text_to_anonymize = request.POST.get('text_to_anonymize')
            anonymized_text = text_to_anonymize
            tf_idf_scores = tf_idf.obtain_tf_idf_scores(
                user.id, text_to_anonymize)
            # Threshold is currently hardcoded. Change it to dynamic generation
            # or DB read
            threshold = 0.4
            for token in tf_idf_scores:
                if tf_idf_scores[token] > threshold:
                    anonymized_text = re.sub(
                        token, 'REDACTED', anonymized_text, flags=re.I)
            return render(request, 'tf_idf_anonymize.html', {'anonymized_text': anonymized_text, 'show_output': True, 'text_to_anonymize': text_to_anonymize, 'threshold': threshold})
        else:
            return render(request, 'tf_idf_anonymize.html')

    else:
        return HttpResponseRedirect('/login')


def token_level_tf_idf_anonymize(response_dict, user):
    ''' Function takes the response dict from NER anonymization and applies TF-IDF detection and anonymizatio'''
    text_to_anonymize = response_dict['original_text']
    tf_idf_scores = tf_idf.obtain_tf_idf_scores(
        user.id, text_to_anonymize)
    response = response_dict['response']
    try:
        tf_idf_configuration = TF_IDF_configuration.objects.get(user=user)
        replacement = tf_idf_configuration.replacement
        threshold = tf_idf_configuration.threshold
    except TF_IDF_configuration.DoesNotExist:
        # In case the configuration does not exist, revert to default values
        replacement = 'REDACTED'
        threshold = 0.4
    for index, entry in enumerate(response):
        # Checking if the token has been assigned a replacement
        if not entry['is_replaced']:
            try:
                if tf_idf_scores[entry['token'].lower()] > threshold:
                    response_dict['response'][index][
                        'replacement'] = replacement
            except KeyError:
                # When the token is non existant in TF_IDF, assumed to be rare
                if entry['token'] not in stop_words:
                    # except when token is a stop_word
                    response_dict['response'][index][
                        'replacement'] = replacement
    return response_dict


def upload_file_to_knowledgebase(request):
    if request.user.is_authenticated:
        user = request.user
        user_id = user.id
        if request.method == 'POST' and request.FILES['myfile']:
            myfile = request.FILES['myfile']
            file_text_lines = [line.decode('utf8').strip()
                               for line in myfile.readlines(
            )]
            file_text = ' '.join(file_text_lines)
            tf_idf.generate_idf_counts(user_id, file_text)
            return render(request, 'upload_file_to_knowledgebase.html')
        return render(request, 'upload_file_to_knowledgebase.html')
    else:
        return HttpResponseRedirect('/login')


@api_view(['POST'])
def upload_file_to_knowledgebase_api(request):
    header_token = request.META.get('HTTP_AUTHORIZATION', None)
    if header_token is not None:
        try:
            token = sub('Token ', '', request.META.get(
                'HTTP_AUTHORIZATION', None))
            token_obj = Token.objects.get(key=token)
            request.user = token_obj.user
            user = request.user
            user_id = user.id
        except Token.DoesNotExist:
            return HttpResponse('INVALID TOKEN')

        myfile = request.FILES['file_to_upload']
        file_text_lines = [line.decode('utf8').strip()
                           for line in myfile.readlines(
        )]
        file_text = ' '.join(file_text_lines)
        tf_idf.generate_idf_counts(user_id, file_text)
        return JsonResponse({'upload_status': 'uploaded succesfully'}, status=200)
    else:
        return HttpResponse("ERROR, ADD A HEADER TOKEN")
