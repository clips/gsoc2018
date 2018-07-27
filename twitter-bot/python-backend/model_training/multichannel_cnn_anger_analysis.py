import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#from keras.wrappers.scikit_learn import KerasClassifier
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils.vis_utils import plot_model
#from keras.metrics import binary_accuracy, categorical_accuracy
from keras import metrics as mets
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from grasp import cd
from grasp import tokenize 
from grasp import chngrams
from grasp import ngrams


##Wrapper methods to use with pandas .apply()
#def _tokenize(text):
#    return tokenize(text)
#    
#def ngramize(text, n = 1):
#    _ngrams = set()
#    _ngrams.update(ngrams(text, n=n))
#    return _ngrams
#    
#def chngramize(text, n = 1):
#    _chngrams = set()
#    _chngrams.update(chngrams(text, n=n))
#    return _chngrams

# Loading and cleaning the data
tweets_data = pd.read_csv(cd('twitter_data.csv'), index_col=False, names=['original_index','tweet_id', 'text', 'topic', 'anger'])
tweets_data.drop(['original_index','tweet_id','topic'], axis=1, inplace=True) 
tweets_data = tweets_data.query('not(anger > 1)')
tweets_data = tweets_data[tweets_data.text.str.contains('I liked a @YouTube') == False] # getting rid of some commonly occuring fluff text
tweets_data = tweets_data.dropna()

# Get tokenized version of the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['text'])
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(tweets_data['text'])
length = 250
padded = pad_sequences(encoded, maxlen=length, padding='post')

# saving tokenize
with open('multichannel_cnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Testing model with encoded data
Y = data.iloc[:,-1]
Y = Y.as_matrix()

# define the model
def define_multichannel_cnn_model(length, vocab_size):
    # channel 1
    inputs_1 = Input(shape=(length,))
    embedding_1 = Embedding(vocab_size, 100)(inputs_1)
    conv_1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding_1)
    dropout_1 = Dropout(0.5)(conv_1)
    pool_1 = MaxPooling1D(pool_size=2)(dropout_1)
    flatten_1 = Flatten()(pool_1)
    # channel 2
    inputs_2 = Input(shape=(length,))
    embedding_2 = Embedding(vocab_size, 100)(inputs_2)
    conv_2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding_2)
    dropout_2 = Dropout(0.5)(conv_2)
    pool_2 = MaxPooling1D(pool_size=2)(dropout_2)
    flatten_2 = Flatten()(pool_2)
    # channel 3
    inputs_3 = Input(shape=(length,))
    embedding_3 = Embedding(vocab_size, 100)(inputs_3)
    conv_3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding_3)
    dropout_3 = Dropout(0.5)(conv_3)
    pool_3 = MaxPooling1D(pool_size=2)(dropout_3)
    flatten_3 = Flatten()(pool_3)
    # merge
    concatenated = concatenate([flatten_1, flatten_2, flatten_3])
    # interpretation
    dense = Dense(10, activation='relu')(concatenated)
    outputs = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Multichannel CNN:\n')
    model.summary()
#	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

X_train, X_test, Y_train, Y_test = train_test_split(padded, Y, test_size=0.2, random_state=42)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

model = define_multichannel_cnn_model(length, vocab_size)

import time
start_time = time.time()
model.fit([X_train, X_train, X_train], Y_train, epochs=10, batch_size=32, validation_data=([X_dev, X_dev, X_dev], Y_dev))
end_time = time.time()
training_time = end_time - start_time

print('The training took:', training_time, 'seconds.')
# evaluate model on training dataset
loss, acc = model.evaluate([X_train,X_train,X_train], Y_train, verbose=0)
print('Train Loss: %f' % (loss))
print('Train Accuracy: %f' % (acc*100))
 
# evaluate model on test dataset dataset
loss, acc = model.evaluate([X_test,X_test,X_test], Y_test, verbose=0)
print('Test Loss: %f' % loss)
print('Test Accuracy: %f' % (acc*100))

model.save('multichannel_cnn_model.h5')



#
#
## Splitting into data and labels
#Y = data.iloc[:,-1]
#X = data.drop(data.columns[-1], axis = 1)
#
## Transform X into respective ngrams an chngrams
#X['text'] = X['text'].apply(lambda x:_tokenize(x))
#X_1_chngram = X_2_chngram = X_3_chngram = X_1_ngram = X_2_ngram = X
#X_1_chngram['text'] = X_1_chngram['text'].apply(lambda x: chngramize(x, 1))
#X_2_chngram['text'] = X_2_chngram['text'].apply(lambda x: chngramize(x, 2))
#X_3_chngram['text'] = X_3_chngram['text'].apply(lambda x: chngramize(x, 3))
#X_1_ngram['text'] = X_1_ngram['text'].apply(lambda x: ngramize(x, 1))
#X_2_ngram['text'] = X_2_ngram['text'].apply(lambda x: ngramize(x, 2))
#
#
#X_matrix = X.as_matrix()
#X_1_chngram_matrix = X_1_chngram.as_matrix()
#X_2_chngram_matrix = X_2_chngram.as_matrix()
#X_3_chngram_matrix = X_3_chngram.as_matrix()
#X_1_ngram_matrix = X_1_ngram.as_matrix()
#X_2_ngram_matrix = X_2_ngram.as_matrix()
#Y_matrix = Y.as_matrix()
#
#input_shape = X_matrix[1].shape # for Keras input layer
#num_outputs = 1 # binary classification
#
## Splitting data into train, dev and test sets
#X_1_chngram_train, X_1_chngram_test, Y_train, Y_test = train_test_split(X_1_chngram_matrix, Y_matrix, test_size=0.2, random_state=42)
#X_1_chngram_train, X_1_chngram_dev, Y_train, Y_dev = train_test_split(X_1_chngram_train, Y_train, test_size=0.2, random_state=42)
#
#X_2_chngram_train, X_2_chngram_test, Y_train, Y_test = train_test_split(X_2_chngram_matrix, Y_matrix, test_size=0.2, random_state=42)
#X_2_chngram_train, X_2_chngram_dev, Y_train, Y_dev = train_test_split(X_2_chngram_train, Y_train, test_size=0.2, random_state=42)
#
#X_3_chngram_train, X_3_chngram_test, Y_train, Y_test = train_test_split(X_3_chngram_matrix, Y_matrix, test_size=0.2, random_state=42)
#X_3_chngram_train, X_3_chngram_dev, Y_train, Y_dev = train_test_split(X_3_chngram_train, Y_train, test_size=0.2, random_state=42)
#
#X_1_ngram_train, X_1_ngram_test, Y_train, Y_test = train_test_split(X_1_ngram_matrix, Y_matrix, test_size=0.2, random_state=42)
#X_1_ngram_train, X_1_ngram_dev, Y_train, Y_dev = train_test_split(X_1_ngram_train, Y_train, test_size=0.2, random_state=42)
#
#X_2_ngram_train, X_2_ngram_test, Y_train, Y_test = train_test_split(X_2_ngram_matrix, Y_matrix, test_size=0.2, random_state=42)
#X_2_ngram_train, X_2_ngram_dev, Y_train, Y_dev = train_test_split(X_2_ngram_train, Y_train, test_size=0.2, random_state=42)
#
##X_1_chngram_train = X_1_chngram_train.reshape((1, len(X_1_chngram_train)))
#
#input_shape = (X_1_chngram_train.shape[0],)
#
#model = Sequential()
#model.add(Dense(24, activation='relu', input_shape=(X_1_chngram_train.shape[1],), kernel_initializer='uniform'))
#model.add(Dense(24, activation='relu'))
#model.add(Dense(24, activation='relu'))
#model.add(Dense(24, activation='relu'))
#model.add(Dense(num_outputs, activation='sigmoid'))
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#model.fit(X_1_chngram_train, Y_train, epochs=50, validation_data=(X_1_chngram_dev, Y_dev))

                
#
#model = Sequential()
#model.add(Dense(64, activation='relu', input_shape=(X_matrix.shape[1],), kernel_initializer='uniform'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='softmax'))
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#model.fit(X_matrix, Y_matrix, epochs=50)

