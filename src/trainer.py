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
            temp_line = np.zeros(shape=(len(line), self.embedding_dimensions),
                                 dtype=np.float)
            for key, word in enumerate(line):
                if word in self.embedding.dictionary:
                    index = self.embedding.dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                temp_line[key] = self.embedding.embedding[index]
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
                                      self.second_layer_depth,
                                      self.window_size)

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
    """Made for containing the graphs data definitions and general callables."""
    def __init__(self, kernel_sizes, embedding_dimensions,
                 first_layer_depth, second_layer_depth,
                 window_size):
        self.embedding_dimensions = embedding_dimensions
        self.window_size = window_size
        self.kernel_sizes = kernel_sizes
        self.first_layer_depth = first_layer_depth
        self.second_layer_depth = second_layer_depth
        self.combined_layer_depth = second_layer_depth * window_size
        self.conv_layer_1 = None
        self.conv_layer_2 = None
        self.temp_array = None
        self.combined_layer_1 = None
        self.combined_layer_2 = None
        self.flattened_layer_1 = None
        self.hidden_layer = None
        self.weights = tf.Variable(tf.truncated_normal(
            [self.combined_layer_depth, self.embedding_dimensions], stddev=0.1))
        self.biases = tf.Variable(tf.zeros([self.embedding_dimensions]))
        ## Inputs of form [sentence_length + window_size, embedding_dimensions, 1]
        self.input_data = tf.placeholder(tf.float32, shape=[None, embedding_dimensions])
        self.sentence_length = tf.shape(self.input_data)[0]
        self.paddings = [1, 0]
        ## Generate smaller tensors of size [window_size, embedding_dimensions, 1]
        self.input_data_padded = tf.pad(self.input_data, self.paddings, "CONSTANT")
        self.logits = (tf.zeros([0, embedding_dimensions]))
        self.i = tf.constant(0)
        self.condition = lambda i: tf.less(self.i, self.sentence_length)
        self.body = self.body_loop
        ## This is where the logits are made.
        self.looper = tf.while_loop(self.condition, self.body, [self.i])
        ## Followed by the loss calculation.
        self.loss = tf.losses.mean_squared_error(self.input_data, self.logits)
        ## And finally here our optimizer.
        self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

    def body_loop(self):
        """The main body of the graph that will be called to generate output."""
        self.temp_array = tf.slice(self.input_data_padded,
                                   [self.i, 0, 0],
                                   [self.window_size, self.embedding_dimensions, 1])
        self.conv_layer_1 = [ConvAndReLu(self.kernel_sizes['1x1'], self.embedding_dimensions,
                                         self.first_layer_depth, self.temp_array),
                             ConvAndReLu(self.kernel_sizes['2x2'], self.embedding_dimensions,
                                         self.first_layer_depth, self.temp_array),
                             ConvAndReLu(self.kernel_sizes['3x3'], self.embedding_dimensions,
                                         self.first_layer_depth, self.temp_array),
                             ConvAndReLu(self.kernel_sizes['4x4'], self.embedding_dimensions,
                                         self.first_layer_depth, self.temp_array)]

        self.conv_layer_2 = [ConvAndReLu(self.kernel_sizes['1x1'], self.first_layer_depth,
                                         self.second_layer_depth, self.conv_layer_1[0].hidden),
                             ConvAndReLu(self.kernel_sizes['1x1'], self.first_layer_depth,
                                         self.second_layer_depth, self.conv_layer_1[1].hidden),
                             ConvAndReLu(self.kernel_sizes['1x1'], self.first_layer_depth,
                                         self.second_layer_depth, self.conv_layer_1[2].hidden),
                             ConvAndReLu(self.kernel_sizes['1x1'], self.first_layer_depth,
                                         self.second_layer_depth, self.conv_layer_1[3].hidden)]

        ## Instead, lets try just flattening each layer and using that for the combined.
        self.flattened_layer_1 = [tf.reshape(self.conv_layer_2[0], [-1]),
                                  tf.reshape(self.conv_layer_2[1], [-1]),
                                  tf.reshape(self.conv_layer_2[2], [-1]),
                                  tf.reshape(self.conv_layer_2[3], [-1])]

        self.combined_layer_1 = tf.stack([tf.stack(self.flattened_layer_1[:2], axis=0),
                                          tf.stack(self.flattened_layer_1[2:], axis=0)], axis=0)

        ## Max pool here as well from [2, 2, second_layer_depth] to [1, second_layer_depth]
        ## Then add a fully connected layer to go from second_layer_depth to
        ## our embedding size.

        self.combined_layer_2 = tf.nn.max_pool(self.combined_layer_1,
                                               [2, 2, 1, 1],
                                               [2, 2, 1, 1],
                                               padding='SAME')
        self.hidden_layer = tf.nn.relu(self.weights + self.biases)
        self.logits = tf.concat([self.logits, self.hidden_layer], 0)
        tf.add(self.i, 1)



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
