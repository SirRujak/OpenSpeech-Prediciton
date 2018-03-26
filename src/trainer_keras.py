#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Used to train a neural network to predict words."""

import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import generator
import json
from keras import losses
from keras.layers import Input, Dense, Flatten, Conv1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import keras


from openspeechsetup import *


class DataHolder:
    """A simple holder class for keeping embeddings."""
    def __init__(self, embedding_filename='SK_full_20k_embedding.pickle'):
        with open(embedding_filename, 'rb') as temp_file:
            temp_holder = pickle.load(temp_file)
            self.dictionary = temp_holder.dictionary
            self.reverse_dictionary = temp_holder.reverse_dictionary
            self.embedding = temp_holder.embedding
            ##self.key = temp_holder.key


class BatchGenerator(object):
    """Returns a numpy array representing a sentence of embeddings."""
    def __init__(self, data, train_info, gen_type, data_holder):
        self.data = data
        self.data_length = len(self.data)
        self.num_test_examples = train_info['num_test_examples']
        self.num_val_examples = train_info['num_val_examples']
        self.embedding_dimensions = train_info['embedding_dimensions']
        self.data_holder = data_holder
        self.cursor = 0
        self.gen_type = gen_type
        self.data_holder = data_holder
        self.num_words = len(data_holder.embedding)

    def next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch_header_size = 5
        batch_header = np.zeros((batch_header_size - 1), dtype=np.int)
        while True:
            batch_data = self.data[self.cursor]
            if self.gen_type == "train":
                self.cursor = (self.cursor + 1) % self.data_length
                if self.cursor < self.num_test_examples + self.num_val_examples:
                    self.cursor = self.num_test_examples + self.num_val_examples
            elif self.gen_type == "test":
                self.cursor = (self.cursor + 1) % self.num_test_examples
            else:
                self.cursor = (self.cursor + 1) % self.num_val_examples
                if self.cursor < self.num_test_examples:
                    self.cursor = self.num_test_examples
            batch_len = np.shape(batch_data)[0]
            batch_data = np.concatenate((batch_header, batch_data), axis=0)
            temp_batch_X = []
            temp_batch_Y = []
            for temp_index in range(batch_len):
                feed_input = batch_data[temp_index:temp_index + batch_header_size][:]
                temp_list = []
                #print("Feed input = {}".format(feed_input))
                for key, val in enumerate(feed_input):
                    #print(key, val)
                    temp_list.append(self.data_holder.embedding[val])
                feed_input = np.array(temp_list)
                feed_output1 = batch_data[temp_index + batch_header_size - 1]
                ##feed_output2 = self.data_holder[feed_output1]
                ##print(batch_data[batch_header_size + temp_index])
                #feed_output = np.expand_dims(feed_output1, axis=0)
                temp_batch_X.append(feed_input)
                temp_batch_Y.append(feed_output1)
                #temp_batch_Y.append(np.random.uniform(-1, 1, self.embedding_dimensions))
            batch_X = np.array(temp_batch_X)
            batch_Y = np.array(temp_batch_Y)
            batch_Y = keras.utils.to_categorical(temp_batch_Y, self.num_words)
            ##print("INPUT_SHAPE={}".format(batch_X.shape))
            ##print("OUTPUT_SHAPE={}".format(batch_Y.shape))
            yield batch_X, batch_Y

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
        self.num_val_examples = train_info['num_val_examples']
        self.train_info = train_info
        self.train_info['embedding_dimensions'] = self.embedding_dimensions
        self.test_batches = list()
        self.val_batches = list()
        self.graph = tf.Graph()
        self.steps_to_take = train_info['steps_to_take']
        self.train_graph = None
        self.batch_generator = None
        self.test_generator = None
        self.validation_generator = None
        self.saver = None
        self.data_holder = DataHolder()
        self.num_words = len(self.data_holder.embedding)
        self.setup()
        self.sess = tf.Session(graph=self.graph)
        self.keras_sess = K.set_session(self.sess)

    def setup(self):
        """Creates the graph."""
        if not os.path.exists(self.tf_dataset_filename):
            self.read_data()
            self.maybe_build_dataset()
        else:
            print('Found embedding at %s. Skipping generation. Loading.' % self.tf_dataset_filename)
            with open(self.tf_dataset_filename, 'rb') as temp_file:
                self.train_data = pickle.load(temp_file)
        self.batch_generator = BatchGenerator(self.train_data, self.train_info, "train", self.data_holder)
        self.test_generator = BatchGenerator(self.train_data, self.train_info, "test", self.data_holder)
        self.validation_generator = BatchGenerator(self.train_data, self.train_info, "val", self.data_holder)
        print('Building computational graph.')
        with self.graph.as_default():
            self.generate_graph()
        print('Graph built.')

    def read_data(self):
        """Load the data from each line and put it in a list of lists.
            Each inner list is a sentence."""
        print('Generating list for embedding.')
        for temp_file_name in self.train_data_filenames:
            with open(temp_file_name, 'r', encoding='utf8') as temp_file:
                for line in temp_file:
                    temp_line = line.strip().split()
                    self.word_list.append(temp_line)
        print('List is %d words long.' % len(self.word_list))

    def maybe_build_dataset(self):
        """Converts each word in the data to its numerical representation."""
        print('Building dataset. This may take a while.')
        for line_number, line in enumerate(self.word_list):
            ##temp_line = np.zeros(shape=(len(line), self.embedding_dimensions),
            ##                     dtype=np.float)
            temp_line = []
            for key, word in enumerate(line):
                if word in self.embedding.dictionary:
                    index = self.embedding.dictionary[word]
                else:
                    index = 0  # dictionary['UNK']
                ##temp_line[key] = self.embedding.embedding[index]
                temp_line.append(index)
            temp_line = np.array(temp_line, dtype=np.int)
            self.train_data.append(temp_line)
            if line_number % 100000 == 0:
                print('Processed %d lines.' % line_number)
                print(temp_line)
            if line_number == 200000:
                break
        ## Clearing out the data that won't be used.
        self.word_list = list()
        ## Save the dataset.
        with open(self.tf_dataset_filename, 'wb') as temp_file:
            pickle.dump(self.train_data, temp_file)
        print('Dataset built.')

    def generate_graph(self):
        with self.graph.as_default():
            inputs = Input(shape=(self.window_size, self.embedding_dimensions,), name="input_layer")

            tower_1 = Conv1D(4, (1), padding='same', activation='relu')(inputs)
            tower_1 = Conv1D(4, (3), padding='same', activation='relu')(tower_1)

            tower_2 = Conv1D(4, (1), padding='same', activation='relu')(inputs)
            tower_2 = Conv1D(4, (5), padding='same', activation='relu')(tower_2)

            ##tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
            ##tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
            output = keras.layers.concatenate([tower_1, tower_2], axis=1)
            x = Flatten()(output)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            '''
            x = Flatten()(inputs)
            x = Dense(128, activation='relu')(x)
            '''
            #predictions = Dense(self.embedding_dimensions, name="output_layer")(x)
            predictions = Dense(self.num_words,  name="output_layer", activation='softmax')(x)
            self.model = Model(inputs, predictions, name="main_model")
            #self.model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self):
        ## TODO: Change this to run the keras graph.
        with self.graph.as_default():
            os.chdir('..')
            os.chdir('results')
            if not os.path.exists("input-5-output-10k-model.h5"):
                training_generator = self.batch_generator.next_batch()
                validation_generator = self.validation_generator.next_batch()
                self.model.fit_generator(generator = training_generator,
                            steps_per_epoch = len(self.train_data) - self.num_val_examples - self.num_test_examples,
                            validation_data = validation_generator,
                            validation_steps = self.num_val_examples)


                self.model.save("input-5-output-10k-model.h5")
            else:
                self.model = load_model("input-5-output-10k-model.h5")

            model_name = "input-5-output-10k-model"
            input_node_name = "input_layer"
            output_node_name = "output_layer/Softmax"
            tf.train.Saver().save(K.get_session(), "./" + model_name + ".chkp")
            ##print(K.get_session().graph_def.node)
            tf.train.write_graph(K.get_session().graph_def, './',
                                 model_name + "_graph.pbtx")
            freeze_graph.freeze_graph('./' + model_name + "_graph.pbtx", None,
                                      False, "./" + model_name + ".chkp",
                                      output_node_name, "save/restore_all",
                                      "save/Const:0",
                                      "./frozen_" + model_name + ".pb", True,
                                      "")

            input_graph_def = tf.GraphDef()
            with tf.gfile.Open("./frozen_" + model_name + ".pb", "rb") as f:
                input_graph_def.ParseFromString(f.read())

            output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, [input_node_name], [output_node_name], tf.float32.as_datatype_enum)

            with tf.gfile.FastGFile("./tensorflow_lite" + model_name + ".pb", "wb") as f:
                f.write(output_graph_def.SerializeToString())

            with open("word_to_index_dict.json", 'w') as f:
                json.dump(self.data_holder.dictionary, f)
            with open("index_to_word_dict.json", "w") as f:
                json.dump(self.data_holder.reverse_dictionary, f)


            os.chdir('..')
            os.chdir('Datasets')

    def save_tensorflow(self):
        model_name = "input-5-output-10k-model"
        input_node_name = "input_layer"
        output_node_name = "output_layer"
        tf.train.write_graph(K.get_session().graph_def, 'out', model_name + "_graph.pbtxt")
        tf.train.Saver().save(K.get_session(), "out/" + model_name + ".chkp")
        freeze_graph.freeze_graph('out/' + model_name + "_graph.pbtx", None, False, "out/" + model_name + ".chkp", output_node_name, "save/restore_all", "save/Const:0", "out/frozen_" + model_name + ".pb", True, "")

        input_graph_def = tf.Graph_Def()
        with tf.gfile.Open("out/frozen_" + model_name + ".pb", "rb") as f:
            input_graph_def.ParseFromString(f.read())

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def, [input_node_name], [output_node_name], tf.float32.as_datatype_enum)

        with tf.gfile.FastGFile("out/tensorflow_lite" + model_name + ".pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

def os_setup():
    word2vec_pretrained_link = 'https://docs.google.com/uc?export=download&confirm=iu5Z&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'
    swiftkey_prediction_data_link = 'https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip'
    word2vec_filename = 'GoogleNews-vectors-negative300.bin.gz'
    swiftkey_filename = 'Coursera-SwiftKey.zip'
    dh = DataBuilder(word2vec_pretrained_link, swiftkey_prediction_data_link,
                    word2vec_filename, swiftkey_filename)
    dh.setup()

if __name__ == "__main__":
    os.chdir('..')
    os.chdir('Datasets')
    os_setup()
    TRAINED_EMBEDDING = generator.generate()
    TRAIN_DATA_FILENAMES = ['clean_en_US.blogs.txt',
                            'clean_en_US.news.txt',
                            'clean_en_US.twitter.txt']
    TRAIN_INFO = {'window_size':5,
                  'first_layer_depth':16,
                  'second_layer_depth': 32,
                  'hidden_dimensions': 512,
                  'steps_to_take': 200000,
                  'num_test_examples': 0,
                  'num_val_examples': 80000}
    TRAINER = Trainer(TRAIN_DATA_FILENAMES, TRAINED_EMBEDDING, TRAIN_INFO)
    print(TRAINER.batch_generator.next_batch)
    TRAINER.train()
    ##TRAINER.save_tensorflow()
