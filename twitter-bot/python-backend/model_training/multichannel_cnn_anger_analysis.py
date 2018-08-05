import pickle
import pandas as pd

from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

tweets_data = pd.read_csv('anger_dataset.csv', index_col=False, encoding='unicode_escape')

# Get tokenized version of the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['text'])
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(tweets_data['text'])
length = 250
padded = pad_sequences(encoded, maxlen=length, padding='post')

# saving tokenize
with open('anger_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Testing model with encoded data
Y = tweets_data.iloc[:,-1]
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
    # print('Multichannel CNN:\n')
    # model.summary()
    # plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

X_train, X_test, Y_train, Y_test = train_test_split(padded, Y, test_size=0.1, random_state=42)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

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

model.save('anger_analysis_model.h5')

# Loading and cleaning the data
#tweets_data = pd.read_csv('final_dataset.csv', index_col=False, names=['text', 'topic', 'anger'])
#tweets_data.drop(['topic'], axis=1, inplace=True) 
#tweets_data['anger'] = tweets_data['anger'].replace(2, 1) 
#tweets_data = tweets_data[tweets_data.text.str.contains('I liked a @YouTube') == False] # getting rid of some commonly occuring fluff text
#tweets_data = tweets_data.dropna()
#
#additional_anger_data = pd.read_csv('additional_anger_data.csv', index_col=False, names=['text', 'anger'], encoding='raw_unicode_escape')
#additional_anger_data = additional_anger_data.dropna()
#additional_anger_data = additional_anger_data.query('not(anger == 0)')
#additional_anger_data = additional_anger_data.query('not(anger > 1)')
#
#
#tweets_data = tweets_data.query('not(anger > 1)')
#angry_data = tweets_data.query('not(anger == 0)')
#angry_data = pd.concat([angry_data, additional_anger_data])
#calm_data = tweets_data.query('not(anger == 1)')
#calm_data = calm_data.sample(n=angry_data.shape[0])
#tweets_data = pd.concat([angry_data, calm_data])
