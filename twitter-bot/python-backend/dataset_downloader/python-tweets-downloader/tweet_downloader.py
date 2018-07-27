import sys
import string
import simplejson
from twython import Twython
import datetime
import csv

# import Twitter authentication keys stored elsewhere
from auth import Auth

def download_and_store_tweets(n, topics, result_type = 'mixed', 
                              categories_to_store = ['full_text', 'id_str'], 
                              output_file_path = 'twitter_data', file_type='csv'):
    """ """
    # Get n tweets from Twitter for each topic
    raw_tweets = []
    for topic in topics:
        downloaded_tweets = download_tweets(topic, n, result_type)
        for tweet in downloaded_tweets:
            tweet['topic'] = topic
            tweet['angry'] = 0
            
        raw_tweets = raw_tweets + downloaded_tweets
    
    # filter for categories of interest
    filtered_tweets = []
    for tweet in raw_tweets:
            tweet_data_as_list = []
            for category in categories_to_store:
                tweet_data_as_list.append(tweet[category])
            tweet_data_as_list.append(tweet['topic'])
            tweet_data_as_list.append(tweet['angry'])
            filtered_tweets.append(tweet_data_as_list)
        
#    # Create header for tweets
#    with open(output_file_path + '.' + file_type, 'w', encoding='utf-8') as f:
#        header = ''
#        for category in categories_to_store:
#            header = header + ';' + category
#        header = header + ';topic;angry\n'
#        header = header[1:] # get rid of first comma
#        f.write(header)
    # Now store the data 
    store_tweets(filtered_tweets, output_file_path, file_type)
    return filtered_tweets

def download_tweets(query, n, result_type, filter_retweets = True, filter_replies = True):
    """ """
    if filter_retweets and filter_replies:
        query = query + '-filter:retweets AND -filter:replies'
    else:
        if filter_retweets:
            query = query + '-filter:retweets'
        if filter_replies:
            query = query + '-filter:replies'
        
    raw_data = t.search(q = query, count = n, result_type = result_type, lang = 'en', tweet_mode='extended')
    return raw_data['statuses']

def store_tweets(tweets, output_file_path = 'twitter_data', file_type='csv'):
    """ """
    output_file_path = output_file_path + '.' + file_type
    if file_type is 'txt':
        with open(output_file_path, 'a', encoding='utf-8') as f:
            for i in range(len(tweets)):
                tweet = tweets[i]
                text = ''
                for field in tweet:
                    text = text + ';' + field
                text = text + '\n'
                text = text[1:] # get rid of first comma
                f.write(text);
    elif file_type is 'csv':
        with open(output_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_ALL)
            for i in range(len(tweets)):
                tweet = tweets[i]
                writer.writerow(tweet)            

# create Twython object using authentication values
t = Twython(app_key=Auth['APP_KEY'], 
    app_secret=Auth['APP_SECRET'],
    oauth_token=Auth['OAUTH_TOKEN'],
    oauth_token_secret=Auth['OAUTH_TOKEN_SECRET'])

# specify which fields are of interest for the dataset
fields_to_store = ['full_text', 'id_str']
list_of_new_topics = ['deforestation', 'ecology', '\"global warming\"']
#original_topics_to_store = ['politics', 'Trump', '\"European Union\"', 'Brexit', 'parliament', 'government', 'president', '\"Prime Minister\"',
#                            'sport', 'football', 'hockey', 'running', 'cycling', 
#                            'food', 'vegan', 'cooking', 'nutrition', '\"healthy eating\"',
#                            'education', 'university', 'school', 'teacher', 'exams', 'studying',
#                            'ecology', 'environment', 'sustainability', '\"global warming\"', 'deforestation', 'extinction', '\"plastic ocean\"', 'recycling',
#                            'war', 'conflict', '\"nuclear weapons\"', 'soldiers', 'army', 'military',
#                            'health', 'disease', 'illness', 'death',
#                            '\"social media\"', 'fake news', 
#                            '\"social issues\"', 'poverty', 'racism', 'homophobia', 'injustice' 'hate', 'feminism', 'terrorism',
#                            'culture', 'music', 'movie', 'literature', 'art',
#                            'technolog', 'software', 'hardware', 'computer', '\"artificial intelligence\"',
#                            'nature', 'mountain', 'ocean', 'forest']

#extended_topics_to_store = ['health', 'disease', 'illness', 'cancer', 'heart disease', 'obesity', 'healthy', 'death',
#                            'social media', 'twitter', 'facebook', 'fake news', 'virtual reality', 'online communities', 'forums', '???',
#                            'social issues', 'poverty', 'racism', 'homophobia', 'injustice' 'hate', 'feminism', 'terrorism',
#                            'culture', 'music', 'theatre', 'movie', 'literature', 'art', 'dance', 'design',
#                            'technology', 'smartphone', 'pc', 'software', 'hardware', 'machine learning', 'computer', 'artificial intelligence']

#r = t.search(q = 'vegan', count = 200, result_type = 'mixed', lang = 'en', tweet_mode='extended')

tweets = download_and_store_tweets(n = 20, topics = list_of_new_topics, result_type = 'recent', categories_to_store = fields_to_store, output_file_path = 'last_data_downloads')

#raw_data = t.search(q = '\"EU\" -filter:retweets AND -filter:replies', count = 50, result_type = 'recent', lang = 'en', tweet_mode='extended')

import pandas as pd
import re
import numpy as np

from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
#tweets_original = pd.read_csv('twitter_data.csv', delimiter=';')
#tweets_extended  = pd.read_csv('twitter_data.csv', delimiter=';')
colnames=['full_text','id_str', 'topic', 'angry']
tweets_final  = pd.read_csv('last_data_downloads.csv', delimiter=';', names=colnames)


sorted_tweets = tweets_final.sort_values(by=['topic'])

def delete_http(string):
    return re.sub(r'https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)

sorted_tweets['clean_text'] = sorted_tweets['full_text'].apply(lambda x: delete_http(x))
#sorted_tweets_ai_dropped = sorted_tweets[sorted_tweets.topic != 'ai']
sorted_tweets_full_text_dropped = sorted_tweets.drop(['full_text'], axis=1)
sorted_tweets_renamed = sorted_tweets_full_text_dropped.rename(index=str, columns={"clean_text": "text"}) 

#sorted_extended_dupes_text = sorted_tweets_ai_dropped.drop_duplicates(subset=['full_text'])
#sorted_extended_dupes_id = sorted_tweets_ai_dropped.drop_duplicates(subset=['id_str'])

sorted_removed_duplicates = sorted_tweets_renamed.drop_duplicates(subset=['text', 'id_str'])
#sorted_extended.drop(labels='id_str', axis=1, inplace=True)

colnames=['id_str', 'text', 'topic', 'angry']
sorted_reindexed = sorted_removed_duplicates.reindex(columns=colnames)

sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'obesity']
sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'cancer']
sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'weapons']
sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'land']
sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'twitter']
sorted_reindexed = sorted_reindexed[sorted_reindexed.topic != 'facebook']

sorted_reindexed.to_csv('last_data.csv')
#cleaned_tweets_data = sorted_reindexed
#cleaned_tweets_data['Angry'] = 0

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
sorted_reindexed.groupby('topic').text.count().plot.bar(ylim=0)
plt.show()

#fig = plt.figure(figsize=(8,6))
#cleaned_tweets_data.groupby('topic').text.count().plot.bar(ylim=0)
#plt.show()


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(sorted_reindexed.text).toarray()
labels = sorted_reindexed.topic


model = LogisticRegression(random_state=0)
model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, sorted_reindexed.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

conf_mat_sum = np.sum(conf_mat)
correct_pred_sum = np.sum(np.diagonal(conf_mat))
accuracy = (correct_pred_sum / conf_mat_sum ) * 100
print("Accuracy: ", accuracy)

samples = ["I am angry about the United States of America government",
         "Governmental issues do not concern me at all",
         "The situation with food shortage in Africa is frightening",
         "Famine is still a problem in developing countries",
         "We cannot ignore the threat of global warming any longer!",
         "The average temperature is not rising because of humans",
         "Universities are implementing new policies",
         "The tuition fees are ridiculous!",
         "Scientific reasearch attests to climatic changes on this planet",
         "Democracy has failed",
         "Minarchism is my preferred approach to defining state",
         "Being Vegetarian is about compassion",
         "We need more leaders in public institutions",
         "World Cup in Russia proved that it is still a popular game",
         "I am so happy about the results!",
         "That was a beautiful match",
         "New technologies are helping humans to progress",
         "I am looking forward to seeing that beautiful city",
         "Democracy is bad",
         "Democracy sucks"]

#samples = ["I am doing some grocery shopping today",
#           "I stumbled on a chair and broke my leg"]
text_features = tfidf.transform(samples)
predictions = model.predict(text_features)
for text, predicted in zip(samples, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(predicted))
  print("")

#
#import itertools
#
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#
## Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, y_pred)
#np.set_printoptions(precision=2)
#
## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(conf_mat, classes=list(set(labels)),
#                      title='Confusion matrix, without normalization')
#
## Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(conf_mat, classes=list(set(labels)), normalize=True,
#                      title='Normalized confusion matrix')
#
#
