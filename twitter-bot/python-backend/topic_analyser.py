import pickle
import numpy as np

from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Added to suppress warnings from third patry libraries that clutter the output
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

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

def analyse(tweet):
    """Method that loads in tokenizer, label_encoder and model and uses these to make a prediction on a passed in string."""
    # Needs to be passed in as a list 
    tweet = [tweet]
    # loading the Tokenizer
    tokenizer = None
    with open('models/topic_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #loading label_encoder
    label_encoder = None
    with open('models/topic_label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

    vocab_size = len(tokenizer.word_index) + 1
    encoded_tweet = tokenizer.texts_to_sequences(tweet)
    length = 250
    padded_tweet = pad_sequences(encoded_tweet, maxlen=length, padding='post')

    multichannel_cnn = define_multichannel_cnn_model(length, vocab_size)
    multichannel_cnn.load_weights('models/topic_analysis_model.h5')
    # evaluate model on training dataset
    predictions = multichannel_cnn.predict([padded_tweet, padded_tweet, padded_tweet], verbose=1)

#    prediction = np.argmax(predictions, axis=1)
#    prediction = label_encoder.inverse_transform(predictions)

    predictions_dict = {}
    index = 0
    for prediction in predictions[0]:
        label = label_encoder.inverse_transform(index)
        predictions_dict[label] = prediction  
        index = index + 1
        
    return predictions_dict.keys(), predictions_dict.values()
