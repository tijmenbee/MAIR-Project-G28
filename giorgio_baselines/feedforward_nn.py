from typing import List, Tuple

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense


VOCAB_SIZE = 1000
H_LAYER_SIZE = 32


class FeedForwardNN:
    def __init__(self, training_data: List[Tuple[str, str]], dev_data: List[Tuple[str, str]]):
        acts = [act for act, _ in training_data]
        sentences = [sentence for _, sentence in training_data]

        act_mappings = {word: i for i, word in enumerate(set(acts))}
        labels = [act_mappings[act] for act in acts]

        # Roughly 1000 unique words in training set.
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_matrix(sentences, mode='count')

        labels = np.array(labels)
        one_hot_labels = np.zeros((len(labels), len(set(acts))))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1

        model = Sequential()
        model.add(Dense(H_LAYER_SIZE, activation='relu', input_shape=(1000,)))
        model.add(Dense(len(set(acts)), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        dev_acts = [act for act, _ in dev_data]
        dev_sentences = [sentence for _, sentence in dev_data]

        dev_labels = np.array([act_mappings[act] for act in dev_acts])
        dev_one_hot_labels = np.zeros((len(dev_labels), len(set(acts))))
        for i, label in enumerate(dev_labels):
            dev_one_hot_labels[i, label] = 1
        dev_sequences = tokenizer.texts_to_matrix(dev_sentences, mode='count')

        # Train the model
        model.fit(sequences, one_hot_labels, epochs=1, batch_size=2, validation_data=(dev_sequences, dev_labels))

        self.model = model
        self.tokenizer = tokenizer
        self.act_mappings = act_mappings
        self.info = f"vocab size: {VOCAB_SIZE}, hidden layer size: {H_LAYER_SIZE}"

    def predict(self, sentence: str) -> str:
        new_sequence = self.tokenizer.texts_to_matrix([sentence], mode='count')
        predicted_label = self.model.predict(new_sequence).tolist()[0]
        predicted_label = predicted_label.index(max(predicted_label))

        # Convert integer label back to command word
        int_to_act = {i: word for word, i in self.act_mappings.items()}
        predicted_act = int_to_act[predicted_label]

        return predicted_act

    def predict_multiple(self, sentences: list) -> list:
        new_sequences = self.tokenizer.texts_to_matrix(sentences, mode='count').tolist()
        predicted_labels = self.model.predict(new_sequences).tolist()

        # Convert integer labels back to command words
        int_to_act = {i: word for word, i in self.act_mappings.items()}

        predicted_acts = []
        for predicted_label in predicted_labels:
            label_index = predicted_label.index(max(predicted_label))
            predicted_acts.append(int_to_act[label_index])

        return predicted_acts


if __name__ == "__main__":
    from data import train_data

    FeedForwardNN(train_data)