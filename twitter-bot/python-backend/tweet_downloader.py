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
        downloaded_tweets = download_tweets(topic, n, result_type = 'mixed')
        for tweet in downloaded_tweets:
            tweet['topic'] = topic
            
        raw_tweets = raw_tweets + downloaded_tweets
    
    # filter for categories of interest
    filtered_tweets = []
    for tweet in raw_tweets:
            tweet_data_as_list = []
            for category in categories_to_store:
                tweet_data_as_list.append(tweet[category])
            tweet_data_as_list.append(tweet['topic'])
            filtered_tweets.append(tweet_data_as_list)
        
    # Create header for tweets
    with open(output_file_path + '.' + file_type, 'w', encoding='utf-8') as f:
        header = ''
        for category in categories_to_store:
            header = header + ';' + category
        header = header + ';topic\n'
        header = header[1:] # get rid of first comma
        f.write(header)
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
topics_to_store = ['politics', 'trump', 'obama', 'brexit', 'parliament', 'government', 'president', 'prime minister',
                   'sport', 'football', 'hockey', 'tenis', 'cycling', 'running', 'athletics', 'skiing',
                   'food', 'food waste', 'restaurant', 'vegan', 'cooking', 'hunger', 'nutrition', 'healthy eating',
                   'education', 'university', 'school', 'high school', 'teacher', 'student', 'exams', 'studying',
                   'ecology', 'environment', 'sustainability', 'global warming', 'deforestation', 'extinction', 'plastic ocean', 'recycling',
                   'war', 'conflict', 'nuclear weapons', 'weapons', 'soldiers', 'army', 'military', 'missiles']


#r = t.search(q = 'vegan', count = 200, result_type = 'mixed', lang = 'en', tweet_mode='extended')

tweets = download_and_store_tweets(n = 100, topics = topics_to_store, result_type = 'recent', categories_to_store = fields_to_store)

#raw_data = t.search(q = 'missiles -filter:retweets AND -filter:replies', count = 50, result_type = 'recent', lang = 'en', tweet_mode='extended')

