"""Used to train a neural network to predict words."""
#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import tensorflow as tf
import generator

class DataHolder:
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = []
        self.embedding = []
        self.key = []

class Trainer:
    """Main class for generating a NN for predicting a word given the
        previous words fed to it."""
    def __init__(self, train_data_filenames, embedding):
        self.saver = tf.train.Saver()
        self.word_list = list()
        self.train_data_filenames = train_data_filenames
        self.embedding = embedding

    def read_data(self):
        """Load the data from each line and put it in a list."""
        print('Generating list for embedding.')
        for temp_file_name in self.train_data_filenames:
            with open(temp_file_name, 'r') as temp_file:
                for line in temp_file:
                    temp_line = line.strip().split()
                    self.word_list.extend(temp_line)
        print('List is %d words long.' % len(self.word_list))

    def build_dataset(self):
        """Converts each word in the data to its numerical representation."""
        for word in self.word_list:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            self.data.append(index)

    def generate_batch(self):
        """Reads the next set of words, appends WINDOW_SIZE padding
            to the front and returns it."""
        pass

    def generate_graph(self):
        """Builds the tensorflow graph representation. It contains:
            Four concurent convolutions, four deepening convos, a
            combination, and a max pool."""
        pass

if __name__ == "__main__":
    os.chdir('..')
    os.chdir('Datasets')
    TRAINED_EMBEDDING = generator.generate()
    TRAIN_DATA_FILENAMES = ['clean_en_US.blogs.txt',
                            'clean_en_US.news.txt',
                            'clean_en_US.twitter.txt']
    TRAINER = Trainer(TRAIN_DATA_FILENAMES, TRAINED_EMBEDDING)
