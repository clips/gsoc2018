# script for combining two dataset when rebuilding the dataset

import pandas as pd

original_dataset = pd.read_csv('original_dataset.csv', index_col=False, names=['tweet_id', 'text', 'topic', 'anger'], header=None, encoding='unicode_escape')

topics_to_drop = ['mountain', 'ocean', 'forest', 'nature', 'conflict', 'extinction', 'food', 'vegan', 'cooking', 
                  'nutrition', 'hate', '\"healthy eating\"', 'terrorism', 'injustice', 'hardware', 'football', 
                  'hockey', 'cycling', 'running', 'sport', 'recycling', '\"plastic ocean\"', 'army', 'military', 
                  'soldiers', 'war', '\"nuclear weapons\"', 'death', 'illness', 'disease', 'health', '\"social media\"',
                  'university', 'school', 'teacher', 'exams', 'studying', 'education', 'parliament', 'music', 'art', 'culture',
                  'literature', 'movie', 'technolog', 'software', 'computer', '\"artificial intelligence\"', 'ecology', '\"social issues\"',
                  'fake news']

for topic in topics_to_drop:
    original_dataset = original_dataset[original_dataset.topic != topic]


original_dataset = original_dataset.replace({
            'topic': {
#                    'technolog': 'technology',
#                    'software':'technology',
#                    'computer':'technology',
#                    '\"artificial intelligence\"':'technology',
            
#                    'music':'culture',
#                    'movie':'culture',
#                    'literature':'culture',
#                    'art':'culture',
            
                    'poverty':'\"social issues\"',
                    'racism':'\"social issues\"',
                    'homophobia':'\"social issues\"',
                    'feminism':'\"social issues\"',
            
                    
                    'environment': 'sustainability',
                    '\"global warming\"': 'sustainability',
                    'deforestation': 'sustainability',
                    'ecology': 'sustainability',
                    },
        })
    


additional_dataset = pd.read_csv('doplnene_additional_data.csv', index_col=False, names=['tweet_id', 'text', 'topic', 'anger'], header=None, encoding='unicode_escape')

merged_dataset = pd.concat([original_dataset, additional_dataset])
merged_dataset = merged_dataset[merged_dataset.text.str.contains('@YouTube') == False] # getting rid of some commonly occuring fluff text
merged_dataset = merged_dataset.dropna()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
merged_dataset.groupby('topic').text.count().plot.bar(ylim=0)
plt.show()

merged_dataset.groupby('topic').text.count()

merged_dataset.to_csv('final_dataset.csv')






### MERGING ADDITIONAL DATA TO DATA THAT WAS POST-DOWNLOADED TO EQUALIZE NUM OF EXAMPLES IN CATEGORIES

additional_dataset = pd.read_csv('additional_dataset.csv', index_col=False, names=['tweet_id', 'text', 'topic', 'anger'], header=None, encoding='unicode_escape')
last_data = pd.read_csv('last_data.csv', index_col=False, names=['tweet_id', 'text', 'topic', 'anger'], header=None, encoding='unicode_escape')
last_data = last_data.replace({
            'topic': {
#                    'technolog': 'technology',
#                    'software':'technology',
#                    'computer':'technology',
#                    '\"artificial intelligence\"':'technology',
            
#                    'music':'culture',
#                    'movie':'culture',
#                    'literature':'culture',
#                    'art':'culture',
            
                    'poverty':'\"social issues\"',
                    'racism':'\"social issues\"',
                    'homophobia':'\"social issues\"',
                    'feminism':'\"social issues\"',
            
                    
                    'environment': 'sustainability',
                    '\"global warming\"': 'sustainability',
                    'deforestation': 'sustainability',
                    'ecology': 'sustainability',
                    },
        })
    
merged_dataset = pd.concat([last_data, additional_dataset])
merged_dataset = merged_dataset[merged_dataset.text.str.contains('@YouTube') == False] # getting rid of some commonly occuring fluff text
merged_dataset = merged_dataset.dropna()

merged_dataset.to_csv('doplnene_additional_data.csv')