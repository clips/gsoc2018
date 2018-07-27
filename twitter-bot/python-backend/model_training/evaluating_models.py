import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
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

# define the model
def define_model_one(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 1:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_two(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 2:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_three(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 3:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_four(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 4:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_five(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 5:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_six(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 6:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_seven(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 7:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_eight(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 8:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

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


# Loading and cleaning the data
normal_data = pd.read_csv(cd('normal.csv'), names=['text'])
normal_data['is_incel'] = 0
incel_data = pd.read_csv(cd('incel.csv'), names=['text'])
incel_data['is_incel'] = 1

data = pd.concat([normal_data, incel_data])
data = data.reset_index(drop=True)
data = data.dropna()

# loading the Tokenizer
tokenizer = None
with open('models/mlp_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(data['text'])
length = 200
padded = pad_sequences(encoded, maxlen=length, padding='post')

# Testing model with encoded data
Y = data.iloc[:,-1]
Y = Y.as_matrix()

X_train, X_test, Y_train, Y_test = train_test_split(padded, Y, test_size=0.2, random_state=42)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


model_one = define_model_one(length, vocab_size)
model_one.load_weights('models/mlp_model_one.h5')

# evaluate model on training dataset
model_one_train_loss, model_one_train_acc = model_one.evaluate(X_train, Y_train, verbose=0)
print('Model One Train Accuracy: %f' % (model_one_train_acc*100))
 # evaluate model on test dataset dataset
model_one_test_loss, model_one_test_acc = model_one.evaluate(X_test, Y_test, verbose=0)
print('Model One Test Accuracy: %f' % (model_one_test_acc*100))


model_two = define_model_two(length, vocab_size)
model_two.load_weights('models/mlp_model_two.h5')
# evaluate model on training dataset
model_two_train_loss, model_two_train_acc = model_two.evaluate(X_train, Y_train, verbose=0)
print('Model Two Train Accuracy: %f' % (model_two_train_acc*100))
 # evaluate model on test dataset dataset
model_two_test_loss, model_two_test_acc = model_two.evaluate(X_test, Y_test, verbose=0)
print('Model Two Test Accuracy: %f' % (model_two_test_acc*100))


model_three = define_model_three(length, vocab_size)
model_three.load_weights('models/mlp_model_three.h5')
# evaluate model on training dataset
model_three_train_loss, model_three_train_acc = model_three.evaluate(X_train, Y_train, verbose=0)
print('Model Three Train Accuracy: %f' % (model_three_train_acc*100))
 # evaluate model on test dataset dataset
model_three_test_loss, model_three_test_acc = model_three.evaluate(X_test, Y_test, verbose=0)
print('Model Three Test Accuracy: %f' % (model_three_test_acc*100))


model_four = define_model_four(length, vocab_size)
model_four.load_weights('models/mlp_model_four.h5')
# evaluate model on training dataset
model_four_train_loss, model_four_train_acc = model_four.evaluate(X_train, Y_train, verbose=0)
print('Model Four Train Accuracy: %f' % (model_four_train_acc*100))
 # evaluate model on test dataset dataset
model_four_test_loss, model_four_test_acc = model_four.evaluate(X_test, Y_test, verbose=0)
print('Model Four Test Accuracy: %f' % (model_four_test_acc*100))


model_five = define_model_five(length, vocab_size)
model_five.load_weights('models/mlp_model_five.h5')
# evaluate model on training dataset
model_five_train_loss, model_five_train_acc = model_five.evaluate(X_train, Y_train, verbose=0)
print('Model Five Train Accuracy: %f' % (model_five_train_acc*100))
 # evaluate model on test dataset dataset
model_five_test_loss, model_five_test_acc = model_five.evaluate(X_test, Y_test, verbose=0)
print('Model Five Test Accuracy: %f' % (model_five_test_acc*100))


model_six = define_model_six(length, vocab_size)
model_six.load_weights('models/mlp_model_six.h5')
# evaluate model on training dataset
model_six_train_loss, model_six_train_acc = model_six.evaluate(X_train, Y_train, verbose=0)
print('Model Six Train Accuracy: %f' % (model_six_train_acc*100))
 # evaluate model on test dataset dataset
model_six_test_loss, model_six_test_acc = model_six.evaluate(X_test, Y_test, verbose=0)
print('Model Six Test Accuracy: %f' % (model_six_test_acc*100))


model_seven = define_model_seven(length, vocab_size)
model_seven.load_weights('models/mlp_model_seven.h5')
# evaluate model on training dataset
model_seven_train_loss, model_seven_train_acc = model_seven.evaluate(X_train, Y_train, verbose=0)
print('Model Seven Train Accuracy: %f' % (model_seven_train_acc*100))
 # evaluate model on test dataset dataset
model_seven_test_loss, model_seven_test_acc = model_seven.evaluate(X_test, Y_test, verbose=0)
print('Model Seven Test Accuracy: %f' % (model_seven_test_acc*100))


model_eight = define_model_eight(length, vocab_size)
model_eight.load_weights('models/mlp_model_eight.h5')
# evaluate model on training dataset
model_eight_train_loss, model_eight_train_acc = model_eight.evaluate(X_train, Y_train, verbose=0)
print('Model Eight Train Accuracy: %f' % (model_eight_train_acc*100))
 # evaluate model on test dataset dataset
model_eight_test_loss, model_eight_test_acc = model_eight.evaluate(X_test, Y_test, verbose=0)
print('Model Eight Test Accuracy: %f' % (model_eight_test_acc*100))


# loading the Tokenizer
tokenizer = None
with open('models/multichannel_cnn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(data['text'])
length = 200
padded = pad_sequences(encoded, maxlen=length, padding='post')

# Testing model with encoded data
Y = data.iloc[:,-1]
Y = Y.as_matrix()

X_train, X_test, Y_train, Y_test = train_test_split(padded, Y, test_size=0.2, random_state=42)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


multichannel_cnn = define_multichannel_cnn_model(length, vocab_size)
multichannel_cnn.load_weights('models/multichannel_cnn_model.h5')
# evaluate model on training dataset
multichannel_train_loss, multichannel_train_acc = multichannel_cnn.evaluate([X_train,X_train,X_train], Y_train, verbose=0)
print('Multichannel CNN Train Loss: %f' % multichannel_train_loss) 
print('Multichannel CNN Train Accuracy: %f' % (multichannel_train_acc*100)) 
# evaluate model on test dataset dataset
multichannel_test_loss, multichannel_test_acc = multichannel_cnn.evaluate([X_test,X_test,X_test], Y_test, verbose=0)
print('Multichannel CNN Test Loss: %f' % multichannel_train_loss) 
print('Multichannel CNN Test Accuracy: %f' % (multichannel_test_acc*100))




train_losses = [model_one_train_loss, model_two_train_loss, model_three_train_loss, model_four_train_loss, 
                model_five_train_loss, model_six_train_loss, model_seven_train_loss, model_eight_train_loss,
                multichannel_train_loss]
train_accuracies = [model_one_train_acc, model_two_train_acc, model_three_train_acc, model_four_train_acc, 
                model_five_train_acc, model_six_train_acc, model_seven_train_acc, model_eight_train_acc,
                multichannel_train_acc]

test_losses = [model_one_test_loss, model_two_test_loss, model_three_test_loss, model_four_test_loss, 
                model_five_test_loss, model_six_test_loss, model_seven_test_loss, model_eight_test_loss,
                multichannel_test_loss]
test_accuracies = [model_one_test_acc, model_two_test_acc, model_three_test_acc, model_four_test_acc, 
                model_five_test_acc, model_six_test_acc, model_seven_test_acc, model_eight_test_acc,
                multichannel_test_acc]

print('Results:\n')
print('(Model_9 is the Multichannel CNN model. Other models are variations of a regular multilayer perceptron model.)\n')

print('Train losses:\n')
counter = 1
best_train_loss = train_losses[0]
best_train_loss_model = 1
for loss in train_losses:
    print('Model_%d Train Loss: %f\n' % (counter, loss))
    if loss < best_train_loss:
        best_train_loss = loss
        best_train_loss_model = counter
    counter = counter+1

print('Train accuracies:\n')
counter = 1
best_train_acc = train_accuracies[0]
best_train_acc_model = 1
for acc in train_accuracies:
    print('Model_%d Train Accuracy: %f\n' % (counter, acc))
    if acc > best_train_acc:
        best_train_acc = acc
        best_train_acc_model = counter
    counter = counter+1

print('Test losses:\n')
counter = 1
best_test_loss = test_losses[0]
best_test_loss_model = 1
for loss in test_losses:
    print('Model_%d test Loss: %f\n' % (counter, loss))
    if loss < best_test_loss:
        best_test_loss = loss
        best_test_loss_model = counter
    counter = counter+1

print('Test accuracies:\n')
counter = 1
best_test_acc = test_accuracies[0]
best_test_acc_model = 1
for acc in test_accuracies:
    print('Model_%d Test Accuracy: %f\n' % (counter, acc))
    if acc > best_test_acc:
        best_test_acc = acc
        best_test_acc_model = counter
    counter = counter+1

print('Best train loss: %f achieved with Model_%d\n' % (best_train_loss, best_train_loss_model))
print('Best train accuracy: %f achieved with Model_%d\n' % (best_train_acc, best_train_acc_model))
print('Best test loss: %f achieved with Model_%d\n' % (best_test_loss, best_test_loss_model))
print('Best test accuracy: %f achieved with Model_%d\n' % (best_test_acc, best_test_acc_model))

#from grasp import chngrams
#from grasp import ngrams
#
#
#def ngramize(s,n):
#    return list(ngrams(s,n))
#testing = X_test['text'].apply(lambda x: ngramize(str(x), 2))
#
#testing = []
#
#for line in padded:
#    testing.append(ngramize(line,2))
