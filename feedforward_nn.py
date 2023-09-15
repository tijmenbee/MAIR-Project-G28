from typing import List, Tuple

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense

from data import ACTS


VOCAB_SIZE = 1000
H_LAYER_SIZE = 128
BATCH_SIZE = 5


class FeedForwardNN:
    def __init__(self, training_data: List[Tuple[str, str]], dev_data: List[Tuple[str, str]], epochs=2):
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

        # TODO add more layers? dropout?
        model = Sequential()
        model.add(Dense(H_LAYER_SIZE, activation='relu', input_shape=(VOCAB_SIZE,)))
        model.add(Dense(H_LAYER_SIZE, activation='relu'))
        model.add(Dense(len(ACTS), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        dev_acts = [act for act, _ in dev_data]
        dev_sentences = [sentence for _, sentence in dev_data]

        dev_labels = np.array([act_mappings[act] for act in dev_acts])
        dev_one_hot_labels = np.zeros((len(dev_labels), len(ACTS)))
        for i, label in enumerate(dev_labels):
            dev_one_hot_labels[i, label] = 1
        dev_sequences = tokenizer.texts_to_matrix(dev_sentences, mode='count')

        # Train the model
        history = model.fit(sequences, one_hot_labels, epochs=epochs, batch_size=BATCH_SIZE,
                            validation_data=(
                                dev_sequences,
                                dev_one_hot_labels)
                            )

        self.model = model
        self.tokenizer = tokenizer
        self.act_mappings = act_mappings

        train_accuracy = history.history['accuracy'][-1]
        dev_accuracy = history.history['val_accuracy'][-1]

        self.info = (f"train acc: {train_accuracy:.2f}, dev acc: {dev_accuracy:.2f}, batch: {BATCH_SIZE}, "
                     f"hidden size: {H_LAYER_SIZE}, epochs: {epochs}, vocab size: {VOCAB_SIZE}")

    def predict(self, sentences: List[str]) -> List[str]:
        new_sequences = self.tokenizer.texts_to_matrix(sentences, mode='count')
        predicted_labels = self.model.predict(new_sequences).tolist()

        int_to_act = {i: word for word, i in self.act_mappings.items()}

        predicted_acts = []
        for predicted_label in predicted_labels:
            label_index = predicted_label.index(max(predicted_label))
            predicted_acts.append(int_to_act[label_index])

        return predicted_acts


if __name__ == "__main__":
    from data import train_data, dev_data as d_data

    FeedForwardNN(train_data, d_data)
