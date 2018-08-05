import pickle
import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, Flatten, 
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

# Loading and cleaning the data
tweets_data = pd.read_csv('final_dataset.csv', index_col=False, names=['text', 'topic', 'anger'], encoding = 'unicode-escape')
tweets_data.drop(['anger'], axis=1, inplace=True) 
tweets_data = tweets_data.dropna()

# Get tokenized version of the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['text'])
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(tweets_data['text'])
length = 250
padded = pad_sequences(encoded, maxlen=length, padding='post')
# saving tokenize
with open('topic_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Testing model with encoded data
Y = tweets_data.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode class values as integers
label_encoder = LabelEncoder()
label_encoder.fit(Y)
encoded_Y = label_encoder.transform(Y)

# saving tokenize
with open('topic_label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(encoded_Y)


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
    outputs = Dense(9, activation='softmax')(dense)
    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
model.fit([X_train, X_train, X_train], Y_train, epochs=20, batch_size=32, validation_data=([X_dev, X_dev, X_dev], Y_dev))
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

model.save('topic_analysis_model.h5')

# manual model testing
encoded_test_input = tokenizer.texts_to_sequences(['This is a sample text to predict topic and anger level of. Should be a string in a list.'])
test_input = pad_sequences(encoded_test_input, maxlen=length, padding='post')

predictions = model.predict([test_input, test_input, test_input])

prediction = np.argmax(predictions, axis=1)
prediction = label_encoder.inverse_transform(prediction)
print(prediction)


predictions = model.predict([X_test,X_test,X_test])

predictions_dict = {}
index = 0
for prediction in predictions[0]:
    label = label_encoder.inverse_transform(index)
    predictions_dict[label] = prediction  
    index = index + 1
    
prediction = np.argmax(predictions, axis=1)
prediction = label_encoder.inverse_transform(prediction)
print(prediction)