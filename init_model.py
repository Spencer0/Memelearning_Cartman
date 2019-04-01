from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io


# Open up the training data.
with io.open('C:\\Users\Spencer\\Desktop\\RepSpencer\\SouthParkScriptCrawler\\cartman_data.txt', encoding='utf-8') as f:
    text = f.read().lower()

# Print the length of the data to train with
print('corpus length:', len(text))

# Find the unique chars in our dataset
chars = sorted(list(set(text)))
print('total chars:', len(chars))

# Create a hashtable <Char, Index> that looks something like a:1, b:2
char_indices = dict((c, i) for i, c in enumerate(chars))

# Create a hashtable <index, char> that looks something like 1:a, 2:b
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# Semiredundant meaning we only walk forward 3 chars to start a new 40 length sentence.
maxlen = 50
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# Print the amount of sentences used to train our dataset
print('nb sequences:', len(sentences))

# "Does this sentence contain this character at this position"
# "Does this sentence contian this character"
#Initalize x matrix as (x,y,x) (bool) [everysenctence][sentence is 40 long][each char slot in sentence has uniquechar amount]
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)

#Initialize y matrix as (x,y) (bool) [everysentence][allchars]
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# Answer the above questions
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Save my answers
np.save('x_sample_weights', x)
np.save('y_sample_weights', y)

# build the model: a single LSTM that deals with our input shape
# Input shape = (40, uniquechars) ([particular sentence][what chars it has])
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# LSTM = Long short term memory model type -> remembers stuff (nerds only) ??
# Dense = Type of matrix ??
# Softmax = also not sure

# Create a keras optimaizer and set the learning rate to a low number
# Why? what does a big number do?
# Upon reading the docks, we are 10x the default learning rate
# need to learn more about ML before this makes any sense
optimizer = RMSprop(lr=0.01)

# build and initalize our model
# Categorical crossentropy is something to do with the fact we have as many classes
# to classify into as we have chars (more than 2)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x, y,
          batch_size=128,
          epochs=1)

# Stick my model in a JSON file
# We will also need to save the weights
json_model = model.to_json()
with open('season1_trained_model.json', 'w') as json_file:
    json_file.write(json_model)

model.save_weights("season1_weights.H5")

