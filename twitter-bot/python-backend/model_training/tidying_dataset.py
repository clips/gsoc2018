import pandas as pd

# Loading and cleaning the data
tweets_data = pd.read_csv('twitter_data.csv', index_col=False, names=['original_index','tweet_id', 'text', 'topic', 'anger'])
tweets_data = tweets_data[tweets_data.text.str.contains('@YouTube') == False] # getting rid of some commonly occuring fluff text
tweets_data = tweets_data.query('not(anger > 2)')
tweets_data = tweets_data.dropna()

topics_to_drop = ['mountain', 'ocean', 'forest', 'nature', 'conflict', 'extinction', 'food', 'vegan', 'cooking', 
                  'nutrition', 'hate', '\"healthy eating\"', 'terrorism', 'injustice', 'hardware', 'football', 
                  'hockey', 'cycling', 'running', 'sport', 'recycling', '\"plastic ocean\"', 'army', 'military', 
                  'soldiers', 'war', '\"nuclear weapons\"', 'death', 'illness', 'disease', 'health', '\"social media\"',
                  'university', 'school', 'teacher', 'exams', 'studying', 'education', 'parliament', 'music', 'art', 'culture',
                  'literature', 'movie', 'technolog', 'software', 'computer', '\"artificial intelligence\"', 'ecology']

for topic in topics_to_drop:
    tweets_data = tweets_data[tweets_data.topic != topic]


tweets_data = tweets_data.replace({
            'topic': {
#                    'technolog': 'technology',
#                    'software':'technology',
#                    'computer':'technology',
#                    '\"artificial intelligence\"':'technology',
#                    'music':'culture',
#                    'movie':'culture',
#                    'literature':'culture',
#                    'art':'culture',
#                    'poverty':'\"social issues\"',
#                    'racism':'\"social issues\"',
#                    'homophobia':'\"social issues\"',
#                    'feminism':'\"social issues\"',
            
                    
#                    'environment': 'sustainability',
#                    '\"global warming\"': 'sustainability',
#                    'deforestation': 'sustainability',
#                    'ecology': 'sustainability',
                    },
        })
    
    
list_of_new_topics = ['politics', 'Trump', '\"European Union\"', 'Brexit', 'government', 'president', '\"Prime Minister\"',
                      'sustainability', 'fake news', '\"social issues\"', 'culture', 'technology']
    
tweets_data.to_csv('monday_modified_data3.csv')