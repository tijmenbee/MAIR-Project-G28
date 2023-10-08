from typing import List, Tuple
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow debug stuff

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

from data.data_processor import ACTS


VOCAB_SIZE = 1000
H_LAYER_SIZE = 128
BATCH_SIZE = 5


class FeedForwardNN:
    def __init__(self, training_data: List[Tuple[str, str]], dev_data: List[Tuple[str, str]] = None, epochs=2,
                 debug=False):
        print("Training neural network...")

        self.verbose = 1 if debug else 0

        acts = [act for act, _ in training_data]
        sentences = [sentence for _, sentence in training_data]

        act_mappings = {word: i for i, word in enumerate(ACTS)}
        labels = [act_mappings[act] for act in acts]

        # Roughly 1000 unique words in training set.
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_matrix(sentences, mode='count')

        labels = np.array(labels)
        one_hot_labels = np.zeros((len(labels), len(ACTS)))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1

        model = Sequential()
        model.add(Dense(H_LAYER_SIZE, activation='relu', input_shape=(VOCAB_SIZE,)))
        model.add(Dense(H_LAYER_SIZE, activation='relu'))
        model.add(Dense(len(ACTS), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        validation_data = None
        if dev_data:
            dev_acts = [act for act, _ in dev_data]
            dev_sentences = [sentence for _, sentence in dev_data]

            dev_labels = np.array([act_mappings[act] for act in dev_acts])
            dev_one_hot_labels = np.zeros((len(dev_labels), len(ACTS)))
            for i, label in enumerate(dev_labels):
                dev_one_hot_labels[i, label] = 1
            dev_sequences = tokenizer.texts_to_matrix(dev_sentences, mode='count')

            validation_data = (dev_sequences, dev_one_hot_labels)

        history = model.fit(sequences, one_hot_labels, epochs=epochs, batch_size=BATCH_SIZE,
                            validation_data=validation_data, verbose=self.verbose)
        self.model = model
        self.tokenizer = tokenizer
        self.act_mappings = act_mappings

        train_accuracy = history.history['accuracy'][-1]
        dev_accuracy = 0
        if dev_data:
            dev_accuracy = history.history['val_accuracy'][-1]

        self.info = (f"train acc: {train_accuracy:.2f}, dev acc: {dev_accuracy:.2f}, "
                     f"batch: {BATCH_SIZE}, "
                     f"hidden size: {H_LAYER_SIZE}, epochs: {epochs}, vocab size: {VOCAB_SIZE}")

    def predict(self, sentences: List[str]) -> List[str]:
        new_sequences = self.tokenizer.texts_to_matrix(sentences, mode='count')
        predicted_labels = self.model.predict(new_sequences, verbose=self.verbose).tolist()

        int_to_act = {i: word for word, i in self.act_mappings.items()}

        predicted_acts = []
        for predicted_label in predicted_labels:
            label_index = predicted_label.index(max(predicted_label))
            predicted_acts.append(int_to_act[label_index])

        return predicted_acts
