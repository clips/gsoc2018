import os
import pickle
import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.losses import binary_crossentropy, categorical_crossentropy
#from keras.metrics import binary_accuracy, categorical_accuracy
from keras import metrics as mets
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import top_k_categorical_accuracy

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Loading and cleaning the data
tweets_data = pd.read_csv('final_dataset.csv', index_col=False, names=['tweet_id', 'text', 'topic', 'anger'], encoding = 'unicode_escape')
tweets_data.drop(['tweet_id','anger'], axis=1, inplace=True) 
tweets_data = tweets_data.dropna()

# Get tokenized version of the tweets
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['text'])
vocab_size = len(tokenizer.word_index) + 1
encoded = tokenizer.texts_to_sequences(tweets_data['text'])
length = 250
padded = pad_sequences(encoded, maxlen=length, padding='post')
# saving tokenize
with open('mlp_tokenizer.pickle', 'wb') as handle:
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
with open('mlp_label_encoder.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# convert integers to dummy variables (i.e. one hot encoded)
Y = np_utils.to_categorical(encoded_Y)


# define the model
def define_model_one(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 5:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

# define the model
def define_model_six(length, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=length))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
    model.add(Dense(9, activation='softmax'))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print('Model 8:\n')
    model.summary()
    #	plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

X_train, X_test, Y_train, Y_test = train_test_split(padded, Y, test_size=0.2, random_state=42)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

model_one = define_model_one(length, vocab_size)
model_two = define_model_two(length, vocab_size)
model_three = define_model_three(length, vocab_size)
model_four = define_model_four(length, vocab_size)
model_five = define_model_five(length, vocab_size)
model_six = define_model_six(length, vocab_size)
model_seven = define_model_seven(length, vocab_size)
model_eight = define_model_eight(length, vocab_size)

start_time = time.time()
model_one.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model one took:', training_time, 'seconds.')
# evaluate model on training dataset
model_one_train_loss, model_one_train_acc = model_one.evaluate(X_train, Y_train, verbose=0)
print('Model One Train Accuracy: %f' % (model_one_train_acc*100))
# evaluate model on test dataset dataset
model_one_test_loss, model_one_test_acc = model_one.evaluate(X_test, Y_test, verbose=0)
print('Model One Test Accuracy: %f' % (model_one_test_acc*100))
model_one.save('mlp_topic_model_one.h5')


start_time = time.time()
model_two.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model two took:', training_time, 'seconds.')
# evaluate model on training dataset
model_two_train_loss, model_two_train_acc = model_two.evaluate(X_train, Y_train, verbose=0)
print('Model Two Train Accuracy: %f' % (model_two_train_acc*100))
 # evaluate model on test dataset dataset
model_two_test_loss, model_two_test_acc = model_two.evaluate(X_test, Y_test, verbose=0)
print('Model Two Test Accuracy: %f' % (model_two_test_acc*100))
model_two.save('mlp_topic_model_two.h5')


start_time = time.time()
model_three.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model three took:', training_time, 'seconds.')
# evaluate model on training dataset
model_three_train_loss, model_three_train_acc = model_three.evaluate(X_train, Y_train, verbose=0)
print('Model Three Train Accuracy: %f' % (model_three_train_acc*100))
 # evaluate model on test dataset dataset
model_three_test_loss, model_three_test_acc = model_three.evaluate(X_test, Y_test, verbose=0)
print('Model Three Test Accuracy: %f' % (model_three_test_acc*100))
model_three.save('mlp_topic_model_three.h5')


start_time = time.time()
model_four.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model four took:', training_time, 'seconds.')
# evaluate model on training dataset
model_four_train_loss, model_four_train_acc = model_four.evaluate(X_train, Y_train, verbose=0)
print('Model Four Train Accuracy: %f' % (model_four_train_acc*100))
 # evaluate model on test dataset dataset
model_four_test_loss, model_four_test_acc = model_four.evaluate(X_test, Y_test, verbose=0)
print('Model Four Test Accuracy: %f' % (model_four_test_acc*100))
model_four.save('mlp_topic_model_four.h5')


start_time = time.time()
model_five.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model five took:', training_time, 'seconds.')
# evaluate model on training dataset
model_five_train_loss, model_five_train_acc = model_five.evaluate(X_train, Y_train, verbose=0)
print('Model Five Train Accuracy: %f' % (model_five_train_acc*100))
 # evaluate model on test dataset dataset
model_five_test_loss, model_five_test_acc = model_five.evaluate(X_test, Y_test, verbose=0)
print('Model Five Test Accuracy: %f' % (model_five_test_acc*100))
model_five.save('mlp_topic_model_five.h5')


start_time = time.time()
model_six.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model six took:', training_time, 'seconds.')
# evaluate model on training dataset
model_six_train_loss, model_six_train_acc = model_six.evaluate(X_train, Y_train, verbose=0)
print('Model Six Train Accuracy: %f' % (model_six_train_acc*100))
 # evaluate model on test dataset dataset
model_six_test_loss, model_six_test_acc = model_six.evaluate(X_test, Y_test, verbose=0)
print('Model Six Test Accuracy: %f' % (model_six_test_acc*100))
model_six.save('mlp_topic_model_six.h5')


start_time = time.time()
model_seven.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model seven took:', training_time, 'seconds.')
# evaluate model on training dataset
model_seven_train_loss, model_seven_train_acc = model_seven.evaluate(X_train, Y_train, verbose=0)
print('Model Seven Train Accuracy: %f' % (model_seven_train_acc*100))
 # evaluate model on test dataset dataset
model_seven_test_loss, model_seven_test_acc = model_seven.evaluate(X_test, Y_test, verbose=0)
print('Model Seven Test Accuracy: %f' % (model_seven_test_acc*100))
model_seven.save('mlp_topic_model_seven.h5')


start_time = time.time()
model_eight.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_dev, Y_dev))
end_time = time.time()
training_time = end_time - start_time
print('The training of model eight took:', training_time, 'seconds.')
# evaluate model on training dataset
model_eight_train_loss, model_eight_train_acc = model_eight.evaluate(X_train, Y_train, verbose=0)
print('Model Eight Train Accuracy: %f' % (model_eight_train_acc*100))
 # evaluate model on test dataset dataset
model_eight_test_loss, model_eight_test_acc = model_eight.evaluate(X_test, Y_test, verbose=0)
print('Model Eight Test Accuracy: %f' % (model_eight_test_acc*100))
model_eight.save('mlp_topic_model_eight.h5')



train_losses = [model_one_train_loss, model_two_train_loss, model_three_train_loss, model_four_train_loss, 
                model_five_train_loss, model_six_train_loss, model_seven_train_loss, model_eight_train_loss]
train_accuracies = [model_one_train_acc, model_two_train_acc, model_three_train_acc, model_four_train_acc, 
                model_five_train_acc, model_six_train_acc, model_seven_train_acc, model_eight_train_acc]

test_losses = [model_one_test_loss, model_two_test_loss, model_three_test_loss, model_four_test_loss, 
                model_five_test_loss, model_six_test_loss, model_seven_test_loss, model_eight_test_loss]
test_accuracies = [model_one_test_acc, model_two_test_acc, model_three_test_acc, model_four_test_acc, 
                model_five_test_acc, model_six_test_acc, model_seven_test_acc, model_eight_test_acc]


print('Results:\n')

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

l,a=model_one.evaluate(X_test, Y_test, verbose=1)

# manual model testing
encoded_test_input = tokenizer.texts_to_sequences([''])
test_input = pad_sequences(encoded_test_input, maxlen=length, padding='post')

predictions = model_one.predict(X_test)

predictions_dict = {}
index = 0
for prediction in predictions[0]:
    label = encoder.inverse_transform(index)
    predictions_dict[label] = prediction  
    index = index + 1
    
prediction = np.argmax(predictions, axis=1)
prediction = encoder.inverse_transform(prediction)
print(prediction)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, predictions)

tweets_data['topic'].value_counts()