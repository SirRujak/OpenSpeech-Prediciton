"""Used to train a neural network to predict words."""
#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
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
    def __init__(self, train_data_filenames, embedding,
                 train_info, tf_dataset_filename='tf-dataset.pickle'):
        self.tf_dataset_filename = tf_dataset_filename
        self.word_list = list()
        self.train_data_filenames = train_data_filenames
        self.embedding = embedding
        self.embedding_dimensions = len(embedding.embedding[0])
        self.train_data = []
        self.window_size = train_info['window_size']
        self.first_layer_depth = train_info['first_layer_depth']
        self.second_layer_depth = train_info['second_layer_depth']
        self.hidden_dimensions = train_info['hidden_dimensions']
        self.num_test_examples = train_info['num_test_examples']
        self.test_batches = list()
        self.graph = tf.Graph()
        self.steps_to_take = train_info['steps_to_take']
        self.train_graph = None
        self.batch_generator = None
        self.saver = None
        self.setup()
        self.sess = tf.Session(graph=self.graph)

    def setup(self):
        """Creates the graph."""
        if not os.path.exists(self.tf_dataset_filename):
            self.read_data()
            self.maybe_build_dataset()
        else:
            print('Found embedding at %s. Skipping generation. Loading.' % self.tf_dataset_filename)
            with open(self.tf_dataset_filename, 'rb') as temp_file:
                self.train_data = pickle.load(temp_file)
        self.batch_generator = BatchGenerator(self.train_data)
        for _ in range(self.num_test_examples):
            self.test_batches.append(self.batch_generator.next_batch())
        print('Building computational graph.')
        with self.graph.as_default():
            self.generate_graph()
        print('Graph built.')

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

    def maybe_build_dataset(self):
        """Converts each word in the data to its numerical representation."""
        print('Building dataset. This may take a while.')
        for line_number, line in enumerate(self.word_list):
            temp_line = np.zeros(shape=(len(line), self.embedding_dimensions),
                                 dtype=np.float)
            for key, word in enumerate(line):
                if word in self.embedding.dictionary:
                    index = self.embedding.dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                temp_line[key] = self.embedding.embedding[index]
            self.train_data.append(temp_line)
            if line_number % 100000 == 0:
                print('Processed %d lines.' % line_number)
            if line_number == 200000:
                break
        ## Clearing out the data that won't be used.
        self.word_list = list()
        ## Save the dataset.
        with open(self.tf_dataset_filename, 'wb') as temp_file:
            pickle.dump(self.train_data, temp_file)
        print('Dataset built.')

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
                                      self.window_size,
                                      self.hidden_dimensions)

    def process_batch(self, batch_header_size,
                      batch_header, batch_data, session):
        batch_len = np.shape(batch_data)[0]
        batch_data = np.concatenate((batch_header, batch_data), axis=0)
        for temp_index in range(batch_len):
            feed_input = batch_data[temp_index:temp_index + batch_header_size][:]
            feed_output1 = batch_data[temp_index + batch_header_size]
            ##print(batch_data[batch_header_size + temp_index])
            feed_output = np.expand_dims(feed_output1, axis=0)
            feed_dict = {}
            feed_dict[self.train_graph.input_data] = feed_input
            feed_dict[self.train_graph.output_data] = feed_output

        _, minibatch_loss, logits = session.run(
            [self.train_graph.optimizer,
             self.train_graph.loss,
             self.train_graph.logits], feed_dict=feed_dict)
        return (minibatch_loss, logits)

    def train(self):
        ##
        with self.sess as session:
            tf.global_variables_initializer().run()
            self.saver = tf.train.Saver()
            avg_error = 0
            batch_header_size = 5
            batch_header = np.zeros((batch_header_size, self.embedding_dimensions))
            for step in range(self.steps_to_take):
                batch_data = self.batch_generator.next_batch()
                ##print('Shape of batch_header: {}'.format(np.shape(batch_header)))
                minibatch_loss, _ = self.process_batch(batch_header_size,
                                                       batch_header,
                                                       batch_data, session)
                avg_error += minibatch_loss

                ##feed_dict = {self.input_data:batch_data}
                ##_, minibatch_loss, predictions = session.run(
                ##   [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if step % 1000 == 0:
                    avg_test_error = 0
                    for test_item in self.test_batches:
                        minibatch_loss, _ = self.process_batch(batch_header_size,
                                                               batch_header,
                                                               test_item, session)
                        avg_test_error += minibatch_loss

                    ##mean_squared_error = ((logits - feed_output) ** 2).mean(axis=None)
                    print('Processed %d lines. Saving model.' % step)
                    ## Append the step number to the checkpoint name:
                    self.saver.save(self.sess, 'Word-Prediction-model', global_step=step)
                    print('Minibatch loss: {} at step {}.'.format(minibatch_loss, step))
                    ##print('Mean squared error: {}'.format(mean_squared_error))
                    print('Average train error: {}'.format(avg_error / 1000))
                    print('Average test error: {}'.format(avg_test_error / self.num_test_examples))
                    avg_error = 0
                    ##print('Prediction: {}'.format(logits))
                    ##print('Reality: {}'.format(feed_dict[self.train_graph.output_data]))
                    ##print('Output: {}'.format(feed_output))
                    ##print('Batch data: {}'.format(batch_data))

class ConvAndReLu:
    """Contains all of the steps for one convolution with ReLu."""
    def __init__(self, kernel_size, embedding_dimensions,
                 final_layer_depth, input_data):
        self.weights = tf.Variable(tf.truncated_normal(
            [1, kernel_size, embedding_dimensions, final_layer_depth]))
        self.biases = tf.Variable(tf.zeros([final_layer_depth]))
        self.convolution = tf.nn.conv2d(input_data,
                                        self.weights, [1, 1, 1, 1], padding='SAME')
        self.hidden = tf.nn.relu(self.convolution + self.biases)

class TrainGraph:
    """Made for containing the graphs data definitions and general callables."""
    def __init__(self, kernel_sizes, embedding_dimensions,
                 first_layer_depth, second_layer_depth,
                 window_size, hidden_dimensions):
        self.embedding_dimensions = embedding_dimensions
        self.window_size = window_size
        self.kernel_sizes = kernel_sizes
        self.first_layer_depth = first_layer_depth
        self.second_layer_depth = second_layer_depth
        self.combined_layer_depth = second_layer_depth * window_size
        self.hidden_dimensions = hidden_dimensions
        self.hidden_weights = tf.Variable(tf.truncated_normal(
            [self.combined_layer_depth, self.hidden_dimensions], stddev=0.1))
        self.hidden_biases = tf.Variable(tf.zeros([self.hidden_dimensions]))
        self.weights = tf.Variable(tf.truncated_normal(
            [self.hidden_dimensions, self.embedding_dimensions], stddev=0.1))
        self.biases = tf.Variable(tf.zeros([self.embedding_dimensions]))
        ## Inputs of form [sentence_length + window_size, embedding_dimensions, 1]
        self.input_data = tf.placeholder(tf.float32,
                                         shape=[window_size, embedding_dimensions],
                                         name="input_data")
        self.output_data = tf.placeholder(tf.float32,
                                          shape=[1, self.embedding_dimensions],
                                          name="output_data")
        ##self.sentence_length = tf.shape(self.input_data)[0]
        ##self.paddings = [1, 0]
        ##self.padding = tf.zeros([self.window_size, self.embedding_dimensions])
        ## Generate smaller tensors of size [window_size, embedding_dimensions, 1]
        ##self.input_data_padded = tf.pad(self.input_data, self.paddings, "CONSTANT")
        ##self.input_data_padded = tf.concat([self.padding, self.input_data], 0)
        self.logits = (tf.zeros([1, embedding_dimensions]))
        ##self.i = tf.constant(0)
        ##self.condition = lambda i: tf.less(self.i, self.sentence_length)
        self.body_loop()
        ## This is where the logits are made.
        ##self.looper = tf.while_loop(self.condition, self.body, [self.i])
        ## Followed by the loss calculation.
        self.loss = tf.losses.mean_squared_error(self.output_data, self.logits)
        ##self.loss = tf.losses.mean_pairwise_squared_error(self.output_data, self.logits)
        ##self.loss = tf.reduce_mean(
        ##    tf.nn.softmax_cross_entropy_with_logits(labels=self.output_data, logits=self.logits))
        ## And finally here our optimizer.
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def body_loop(self):
        """The main body of the graph that will be called to generate output."""
        self.temp_array = tf.expand_dims(
            tf.expand_dims(self.input_data, 0), 0)
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
        self.flattened_layer_1 = [tf.reshape(self.conv_layer_2[0].hidden, [-1]),
                                  tf.reshape(self.conv_layer_2[1].hidden, [-1]),
                                  tf.reshape(self.conv_layer_2[2].hidden, [-1]),
                                  tf.reshape(self.conv_layer_2[3].hidden, [-1])]

        self.combined_layer_1 = tf.expand_dims(
            tf.stack([tf.stack(self.flattened_layer_1[:2], axis=0),
                      tf.stack(self.flattened_layer_1[2:], axis=0)], axis=0), 0)

        ## Max pool here as well from [1, 2, 2, second_layer_depth] to [1, second_layer_depth]
        ## Then add a fully connected layer to go from second_layer_depth to
        ## our embedding size.


        self.combined_layer_2 = tf.nn.max_pool(self.combined_layer_1,
                                               [1, 2, 2, 1],
                                               [1, 2, 2, 1],
                                               padding='SAME')
        print(tf.shape(self.combined_layer_2))
        self.squeezed_layer = tf.squeeze(self.combined_layer_2, [0, 1])
        self.hidden_layer = tf.nn.relu(tf.matmul(
            self.squeezed_layer, self.hidden_weights) + self.hidden_biases, name="Logits")
        self.logits = tf.matmul(self.hidden_layer, self.weights) + self.biases
        ##self.logits = tf.concat([self.logits, self.hidden_layer], 0)
        ##tf.add(self.i, 1)



if __name__ == "__main__":
    os.chdir('..')
    os.chdir('Datasets')
    TRAINED_EMBEDDING = generator.generate()
    TRAIN_DATA_FILENAMES = ['clean_en_US.blogs.txt',
                            'clean_en_US.news.txt',
                            'clean_en_US.twitter.txt']
    TRAIN_INFO = {'window_size':5,
                  'first_layer_depth':16,
                  'second_layer_depth': 32,
                  'hidden_dimensions': 512,
                  'steps_to_take': 200000,
                  'num_test_examples': 10000}
    TRAINER = Trainer(TRAIN_DATA_FILENAMES, TRAINED_EMBEDDING, TRAIN_INFO)
    print(TRAINER.batch_generator.next_batch)
    TRAINER.train()
