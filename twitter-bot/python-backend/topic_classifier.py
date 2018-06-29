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


twitter_data = pd.read_csv('twitter_data.csv', delimiter=';')
twitter_data.drop(labels='id_str', axis=1, inplace=True)

#import matplotlib.pyplot as plt
#fig = plt.figure(figsize=(8,6))
#twitter_data.groupby('topic').full_text.count().plot.bar(ylim=0)
#plt.show()

# Get rid of the HTTP links - #text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
for i in range(twitter_data.shape[0]):
    replaced_text = re.sub(r'https?:\/\/.*[\r\n]*', '', twitter_data.iloc[i]['full_text'], flags=re.MULTILINE)
    twitter_data.iloc[i]['full_text'] = replaced_text

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(twitter_data.full_text).toarray()
labels = twitter_data.topic

# Code for making n-grams  
  
#N = 2
#for t in list(set(sorted(twitter_data.topic))):
#  features_chi2 = chi2(features, labels == t)
#  indices = np.argsort(features_chi2[0])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#  print("# '{}':".format(t))
#  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
#  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
#  

# Code for determining the best model to use with data

#models = [
#    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#    LinearSVC(),
#    MultinomialNB(),
#    LogisticRegression(random_state=0),
#]
#CV = 5
#cv_df = pd.DataFrame(index=range(CV * len(models)))
#entries = []
#for model in models:
#  model_name = model.__class__.__name__
#  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#  for fold_idx, accuracy in enumerate(accuracies):
#    entries.append((model_name, fold_idx, accuracy))
#cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
#import seaborn as sns
#
#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
#
#cv_df.groupby('model_name').accuracy.mean()


#model = LinearSVC()
#
#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, twitter_data.index, test_size=0.33, random_state=0)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#
#from sklearn.metrics import confusion_matrix
#conf_mat = confusion_matrix(y_test, y_pred)

model = LogisticRegression(random_state=0)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, twitter_data.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

samples = ["I am angry about the United States of America government",
         "Governmental issues do not concern me at all"
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
         "Ronaldo does know how to kick a ball",
         "Ronaldo scored a great goal",
         "Deesclation bot will probably help the world a lot"]
text_features = tfidf.transform(samples)
predictions = model.predict(text_features)
for text, predicted in zip(samples, predictions):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(predicted))
  print("")
  
