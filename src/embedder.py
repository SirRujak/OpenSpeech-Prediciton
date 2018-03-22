#!/usr/bin/python
# -*- coding: utf-8 -*-
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from itertools import compress
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

class Embedder:
    def __init__(self,
                 filenames=['clean_en_US.blogs.txt',
                            'clean_en_US.news.txt',
                            'clean_en_US.twitter.txt'],
                 vocabulary_size=10000,
                 data_index=0,
                 num_skips=1,
                 skip_window=6,
                 batch_size=1024,
                 embedding_size=28,
                 valid_size=8,
                 valid_window=100,
                 num_sampled=4096,
                 num_steps=3001
                ):
        self.filenames = filenames
        self.word_list = []
        self.vocabulary_size = vocabulary_size
        self.dictionary = dict()
        self.data = list()
        self.count = [['UNK', -1]]
        self.reverse_dictionary = None
        self.data_index = data_index
        self.num_skips = num_skips # How many times to reuse an input to generate a label.
        self.skip_window = skip_window # How many words to consider left and right.
        self.batch_size = batch_size
        self.embedding_size = embedding_size # Dimension of the embedding vector.
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = valid_size # Random set of words to evaluate similarity on.
        self.valid_window = valid_window # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.array(random.sample(range(valid_window), valid_size))
        self.num_sampled = num_sampled # Number of negative examples to sample.
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        self.graph = tf.Graph()
        self.num_steps = num_steps

    def build_embedding(self):
        self.read_data()
        self.build_dataset()
        ##self.test_data()
        return self.train_data()

    def read_data(self):
        """Load the data from each line and put it in a list."""
        print('Generating list for embedding.')
        for temp_file_name in self.filenames:
            with open(temp_file_name, 'r', encoding="utf8") as temp_file:
                for line in temp_file:
                    temp_line = line.strip().split()
                    self.word_list.extend(temp_line)
        print('List is %d words long.' % len(self.word_list))


    def build_dataset(self):
        self.count.extend(collections.Counter(self.word_list).most_common(self.vocabulary_size - 1))
        for word, _ in self.count:
            self.dictionary[word] = len(self.dictionary)
        unk_count = 0
        for word in self.word_list:
            if word in self.dictionary:
                index = self.dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            self.data.append(index)
        self.count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        print('Most common words (+UNK)', self.count[:5])
        print('Sample data', self.data[:10])
        self.word_list = None
        ##return data, count, dictionary, reverse_dictionary

    def generate_batch(self):
        ##global data_index
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)
        return batch, labels

    def test_data(self):

        print('data:', [self.reverse_dictionary[di] for di in self.data[:8]])

        for num_skips, skip_window in [(2, 1), (4, 2)]:
            data_index = 0
            batch, labels = self.generate_batch()
            print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
            print('    batch:', [self.reverse_dictionary[bi] for bi in batch])
            print('    labels:', [self.reverse_dictionary[li] for li in labels.reshape(self.batch_size)])

    """==============================PROGRESS=============================="""

    def create_graph(self):
        with self.graph.as_default(), tf.device('/cpu:0'):

            # Input data.
            self.train_dataset = tf.placeholder(shape=[self.batch_size], dtype=tf.int32)
            self.train_labels = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32)
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Variables.
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            self.softmax_weights = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            self.softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_dataset)
            ##print(tf.DType.is_floating(self.embed))
            ##self.embed = tf.nn.embedding_lookup(self.train_dataset, self.embeddings)
            # Compute the softmax loss, using a sample of the negative labels each time.
            ##self.loss = tf.reduce_mean(
            ##    tf.nn.sampled_softmax_loss(self.softmax_weights,
            ##                               self.softmax_biases,
            ##                               self.train_labels,
            ##                               self.embed,
            ##                               self.num_sampled,
            ##
            ##                               self.vocabulary_size))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.softmax_weights,
                                                      self.softmax_biases,
                                                      self.train_labels,
                                                      self.embed,
                                                      self.num_sampled,
                                                      self.vocabulary_size))

            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
            self.normalized_embeddings = self.embeddings / self.norm
            self.valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(self.valid_embeddings,
                                        tf.transpose(self.normalized_embeddings))

    def run_graph(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            with tf.Session(graph=self.graph) as session:
                tf.global_variables_initializer().run()
                print('Initialized')
                average_loss = 0
                for step in range(self.num_steps):
                    batch_data, batch_labels = self.generate_batch()
                    feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
                    _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    average_loss += l
                    if step % 1000 == 0:
                        if step > 0:
                            average_loss = average_loss / 1000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step %d: %f' % (step, average_loss))
                        average_loss = 0
                    # note that this is expensive (~20% slowdown if computed every 500 steps)
                    if step % 2000 == 0:
                        sim = self.similarity.eval()
                        for i in range(self.valid_size):
                            valid_word = self.reverse_dictionary[self.valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self.reverse_dictionary[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)
                return self.normalized_embeddings.eval()
                ##final_embeddings = self.normalized_embeddings.eval()

    def train_data(self):
        self.create_graph()
        return self.run_graph()

if __name__ == "__main__":
    os.chdir('..')
    os.chdir('Datasets')
    temp = Embedder()
    print(temp.build_embedding())
