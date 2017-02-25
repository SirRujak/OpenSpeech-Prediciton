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
        self.train_graph = None

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
        kernel_sizes = {'1x1':1,
                        '2x2':2,
                        '3x3':3,
                        '4x4':4}
        self.train_graph = TrainGraph(kernel_sizes,
                                      self.embedding_dimensions,
                                      self.first_layer_depth,
                                      self.second_layer_depth)

class ConvAndReLu:
    """Contains all of the steps for one convolution with ReLu."""
    def __init__(self, kernel_size, embedding_dimensions,
                 final_layer_depth, input_data):
        self.weights = tf.Variable(tf.truncated_normal(
            [kernel_size, embedding_dimensions, final_layer_depth]))
        self.biases = tf.Variable(tf.zeros([final_layer_depth]))
        self.convolution = tf.nn.conv1d(input_data, self.weights, 1, padding='SAME')
        self.hidden = tf.nn.relu(self.convolution + self.biases)

class TrainGraph:
    def __init__(self, kernel_sizes, embedding_dimensions,
                 first_layer_depth, second_layer_depth):
        ## Inputs of form [sentence_length, window_size, ]
        self.input_data = tf.placeholder(tf.float32, shape=[None, embedding_dimensions])

        self.conv_layer_1 = [ConvAndReLu(kernel_sizes['1x1'], embedding_dimensions,
                                         first_layer_depth, self.input_data),
                             ConvAndReLu(kernel_sizes['2x2'], embedding_dimensions,
                                         first_layer_depth, self.input_data),
                             ConvAndReLu(kernel_sizes['3x3'], embedding_dimensions,
                                         first_layer_depth, self.input_data),
                             ConvAndReLu(kernel_sizes['4x4'], embedding_dimensions,
                                         first_layer_depth, self.input_data)]

        self.conv_layer_2 = [ConvAndReLu(kernel_sizes['1x1'], first_layer_depth,
                                         second_layer_depth, self.conv_layer_1[0].hidden),
                             ConvAndReLu(kernel_sizes['1x1'], first_layer_depth,
                                         second_layer_depth, self.conv_layer_1[1].hidden),
                             ConvAndReLu(kernel_sizes['1x1'], first_layer_depth,
                                         second_layer_depth, self.conv_layer_1[2].hidden),
                             ConvAndReLu(kernel_sizes['1x1'], first_layer_depth,
                                         second_layer_depth, self.conv_layer_1[3].hidden)]
        ## Weights for the first set of convolutions.
        self.weights_1x1 = tf.Variable(tf.truncated_normal(
            [kernel_sizes['1x1'], embedding_dimensions, first_layer_depth]))
        self.weights_2x2 = tf.Variable(tf.truncated_normal(
            [kernel_sizes['2x2'], embedding_dimensions, first_layer_depth]))
        self.weights_3x3 = tf.Variable(tf.truncated_normal(
            [kernel_sizes['3x3'], embedding_dimensions, first_layer_depth]))
        self.weights_4x4 = tf.Variable(tf.truncated_normal(
            [kernel_sizes['4x4'], embedding_dimensions, first_layer_depth]))
        ## Biases for the first set of hidden layers.
        self.biases_1x1 = tf.Variable(tf.zeros([first_layer_depth]))
        self.biases_2x2 = tf.Variable(tf.zeros([first_layer_depth]))
        self.biases_3x3 = tf.Variable(tf.zeros([first_layer_depth]))
        self.biases_4x4 = tf.Variable(tf.zeros([first_layer_depth]))
        ## First set of convolutions.
        self.conv_1x1 = tf.nn.conv1d(self.input_data, self.weights_1x1, 1, padding='SAME')
        self.conv_2x2 = tf.nn.conv1d(self.input_data, self.weights_2x2, 1, padding='SAME')
        self.conv_3x3 = tf.nn.conv1d(self.input_data, self.weights_3x3, 1, padding='SAME')
        self.conv_4x4 = tf.nn.conv1d(self.input_data, self.weights_4x4, 1, padding='SAME')
        ## First hidden layer.
        self.hidden_1x1 = tf.nn.relu(self.conv_1x1 + self.biases_1x1)
        self.hidden_2x2 = tf.nn.relu(self.conv_2x2 + self.biases_2x2)
        self.hidden_3x3 = tf.nn.relu(self.conv_3x3 + self.biases_3x3)
        self.hidden_4x4 = tf.nn.relu(self.conv_4x4 + self.biases_4x4)



if __name__ == "__main__":
    os.chdir('..')
    os.chdir('Datasets')
    TRAINED_EMBEDDING = generator.generate()
    TRAIN_DATA_FILENAMES = ['clean_en_US.blogs.txt',
                            'clean_en_US.news.txt',
                            'clean_en_US.twitter.txt']
    TRAIN_INFO = {'window_size':5,
                  'first_layer_depth':16,
                  'second_layer_depth': 32,}
    TRAINER = Trainer(TRAIN_DATA_FILENAMES, TRAINED_EMBEDDING, TRAIN_INFO)
