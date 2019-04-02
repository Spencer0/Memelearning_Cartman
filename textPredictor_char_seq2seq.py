from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import re
import matplotlib.pyplot as pp
import seaborn
import pandas

class textPredictor:

    ## Load up and compile the model to be used
    def __init__(self):
        # load json and create model
        json_file = open('season1_trained_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights("season1_weights.h5")

        # Load the training data (Compiled initally from text)
        self.x_train = np.load('x_sample_weights.npy')
        self.y_train = np.load('y_sample_weights.npy')

        # Set up an optimizer and compile the model
        optimizer = RMSprop(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # Grab array sizes from data [not a good way to do this]
        with io.open('C:\\Users\Spencer\\Desktop\\RepSpencer\\SouthParkScriptCrawler\\cartman_data.txt', encoding='utf-8') as f:
            text = f.read().lower()
        self.chars = sorted(list(set(text)))
        self.sentenece_length = 50
        # Create a hashtable <Char, Index> that looks something like a:1, b:2
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))

        # Create a hashtable <index, char> that looks something like 1:a, 2:b
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    ## Helper function for predicting the next character
    def __sample__(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # Cast my prediction array [position in sectence][char it has] to np float array
        preds = np.asarray(preds).astype('float64')

        # take the natural log of all my floats (WHY WHY WHY)
        # then divide that by the chosen diversity level
        preds = np.log(preds) / temperature

        # now, exponentional function to undo the log
        exp_preds = np.exp(preds)

        # preds will now be equal to every slot divided by total slots
        preds = exp_preds / np.sum(exp_preds)

        # Use preds to compute a multinomial distribution
        probas = np.random.multinomial(1, preds, 1)

        # Greedy aglorthim here for character prediction,
        # Always grab the most likely character
        # might run into errors with local maxima
        return np.argmax(probas)

    ## Function to improve model performance by calling .fit
    ## Uses the training numpy arrays created in init_model_char_seq2seq.py
    def train(self, verbosity=1):

        # load the true sample weights for each character
        # Start training the data
        # Every epoch, call on epoch end

        self.model.fit(self.x_train, self.y_train,
                  batch_size=128,
                  epochs=1,
                  verbose=verbosity)

        # Stick my model in a JSON file
        # We will also need to save the weights
        json_model = self.model.to_json()
        with open('season1_trained_model.json', 'w') as json_file:
            json_file.write(json_model)

        self.model.save_weights("season1_weights.H5")

    ## Grabs a random sentence from the training data and keeps
    ## moving from there
    def predictor(self, out_len=400, sentence=None):
        with io.open('C:\\Users\Spencer\\Desktop\\RepSpencer\\SouthParkScriptCrawler\\cartman_data.txt', encoding='utf-8') as f:
            text = f.read().lower()
        # Grabs a random sentence (40 length string) from the script)
        if not sentence: start_index = random.randint(0, len(text) - self.sentenece_length - 1)
        # Loop through each diverstiy level
        # Diverstiy :
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('\n\n----- diversity:', diversity, " -----")

            generated = ''
            if not sentence: sentence = text[start_index: start_index + self.sentenece_length]
            generated += sentence
            print('\n----- Generating with seed: "' + sentence + '" -----\n\n')

            # Produce the 400 next most likely characters to follow our seeded sentence
            for i in range(out_len):
                x_pred = np.zeros((1, self.sentenece_length, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.
                # Input my sample matrix for our sentence, then predict a new character
                preds = self.model.predict(x_pred, verbose=0)[0]
                # Send the matrix to my greedy sample helper function
                next_index = self.__sample__(preds, diversity)
                # turn the choosen index into a char
                next_char = self.indices_char[next_index]
                # add it to my generated string
                # For printing pruposes
                generated += next_char
                # add it to my random sentence minus a character
                # This is hte length it will be looking back at any point
                sentence = sentence[1:] + next_char


            print(generated[40:])

    ## loop training function that logs performance over many cycles
    def long_train(self, epocs):
        default_std_out = sys.stdout
        log_file = open("nightlog.txt", "a")
        print("Begin login to output file...")
        sys.stdout = log_file
        for epoch in range(epocs):
                print("Night " + str(epoch))
                self.train(verbosity=2)
                file_name = "backups\cartman_weights_night"+str(epoch)+".H5"
                self.model.save_weights(file_name)
        sys.stdout = default_std_out
        log_file.close()

    ## Evaluate our model (DOENST WORK)
    def test(self, sentence):
        x_pred = np.zeros((1, self.sentenece_length, len(self.chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, self.char_indices[char]] = 1.
        self.model.evaluate(x_pred)

    ## Print some stats of the model
    def stats(self):
        model = Sequential()
        print(self.model.summary())
        print(self.model.name.__str__())

    ## Create a plot given the nightlog produced in long_train()
    def plotlog(self, filename='nightlog.txt'):
        # Set up lists
        nights = []
        loss = []

        # Set up regex matches
        # Regex to grab any integers after "Night"
        re_night_grabber = "Night.(\d+)\n"
        re_loss_grabber = "loss..(\d+.\d+)"

        # Capture matches in logger text
        with open(filename, 'r') as f:
            lines = f.read()
            nights_found = re.findall(re_night_grabber, lines)
            losses_found = re.findall(re_loss_grabber, lines)

        # Check data
        nights = list(range(len(losses_found)))
        losses = list(map(float, losses_found))
        data = {'Epochs': nights, 'Losses': losses}
        dataframe = pandas.DataFrame(data)
        seaborn.set_style('darkgrid')
        graph = seaborn.lineplot(x='Epochs', y='Losses', data=dataframe)
        title = "Double layer LTSM + Double density (softmax, rectifier) - Cartman data - overnight"
        graph.set_title(title)
        pp.savefig(title+'.png')




tp = textPredictor()
#tp.long_train(50)
#tp.train()
#tp.predictor(out_len=1500)
#tp.plotlog()
#tp.stats()
