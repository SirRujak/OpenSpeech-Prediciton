"""Used to train a neural network to predict words."""
#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import generator

class DataHolder:
    """A simple holder class for keeping embeddings."""
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = []
        self.embedding = []
        self.key = []

class BatchGenerator(object):
    """Returns a numpy array representing a sentence of embeddings."""
    def __init__(self, data):
        self.data = data
        self.data_length = len(self.data)
        self.cursor = 0

    def next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        temp_batch = self.data[self.cursor]
        self.cursor = (self.cursor + 1) % self.data_length
        return temp_batch

class Trainer:
    """Main class for generating a NN for predicting a word given the
        previous words fed to it."""
    def __init__(self, train_data_filenames, embedding, train_info):
        self.saver = tf.train.Saver()
        self.word_list = list()
        self.train_data_filenames = train_data_filenames
        self.embedding = embedding
        self.embedding_dimensions = len(embedding.embedding[0])
        self.train_data = []
        self.window_size = train_info['window_size']
        self.first_layer_depth = train_info['first_layer_depth']
        self.second_layer_depth = train_info['second_layer_depth']
        self.batch_generator = BatchGenerator(self.train_data)
        self.graph = tf.Graph()

    def setup(self):
        """Creates the graph."""
        with self.graph.as_default():
            self.generate_graph()

    def read_data(self):
        """Load the data from each line and put it in a list of lists.
            Each inner list is a sentence."""
        print('Generating list for embedding.')
        for temp_file_name in self.train_data_filenames:
            with open(temp_file_name, 'r') as temp_file:
                for line in temp_file:
                    temp_line = line.strip().split()
                    self.word_list.append(temp_line)
        print('List is %d words long.' % len(self.word_list))

    def build_dataset(self):
        """Converts each word in the data to its numerical representation."""
        for line in self.word_list:
            temp_line = np.zeros(shape=(len(line) + self.window_size, self.embedding_dimensions),
                                 dtype=np.float)
            for key, word in enumerate(line):
                if word in self.embedding.dictionary:
                    index = self.embedding.dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                temp_line[key + 5] = self.embedding.embedding[index]
            self.train_data.append(index)
        ## Clearing out the data that won't be used.
        self.word_list = list()

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
    ##TRAIN_WINDOW = 5
    TRAIN_INFO = {'window_size':5,
                  'first_layer_depth':4,
                  'second_layer_depth': 8,}
    TRAINER = Trainer(TRAIN_DATA_FILENAMES, TRAINED_EMBEDDING, TRAIN_INFO)
