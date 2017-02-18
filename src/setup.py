#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import word2vec
import gzip
import shutil
import os
import numpy as np
import random
import re
##import regex
import string
import tensorflow as tf
import tarfile
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import pickle

class DataHolder:
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = []
        self.embedding = []
        self.data = []

class DataBuilder:
    def __init__(self, w2v_url, sk_url,
                 w2v_file_name, sk_file_name,
                 sorted_words_file='sorted_words.pickle',
                 final_dict_file='sorted_dict_100k.pickle',
                 embedding_file='OpenSpeech-Embeddings.pickle'):
        self.w2v_url = w2v_url
        self.sk_url = sk_url
        self.w2v_file_name = w2v_file_name
        self.w2v_file_name_extracted = w2v_file_name[:-3]
        self.sk_file_name = sk_file_name
        self.sk_file_name_extracted = sk_file_name[:-4]
        self.w2v_size = 1647046227
        self.sk_size = 574661177
        self.sorted_words_file = sorted_words_file
        ## FIXME: Really need to move this to the commented line.
        ## It breaks way too often with just a bad apostrophe.
        self.setup_text_replacement()
        self.model = None
        self.final_dict = {}
        self.final_dict_file = final_dict_file
        self.embedding_file = embedding_file

    def setup(self):
        self.downloader()
        self.maybe_extract()
        self.maybe_clean()
        self.maybe_calc_frequencies()
        self.maybe_build_sorted_dict()
        return self.maybe_build_final()

    def maybe_build_final(self):
        if not os.path.exists(self.embedding_file):
            ##print(self.sorted_vals_100k[0])
            if self.model == None:
                self.load_model()
            print('Generating embedding file at %s.' % self.embedding_file)
            temp_dh = DataHolder()
            temp_dh.dictionary['UNK'] = 0
            temp_dh.reverse_dictionary.append('UNK')
            temp_dh.embedding.append(self.model['UNK'])
            for index, values in enumerate(self.sorted_vals_100k):
                if values[0] in self.final_dict:
                    temp_dh.reverse_dictionary.append(values[0])
                    temp_dh.embedding.append(self.model[values[0]])
                    temp_dh.dictionary[values[0]] = len(temp_dh.reverse_dictionary) - 1

            temp_filenames = ['clean_en_US.blogs.txt',
                              'clean_en_US.news.txt',
                              'clean_en_US.twitter.txt']
            for temp_file_name in temp_filenames:
                with open(temp_file_name, 'r') as temp_file:
                    print('Processing %s.' % temp_file_name)
                    for index, line in enumerate(temp_file):
                        temp_dh.data.append([])
                        temp_line = line.strip().split(' ')
                        for _, word in enumerate(temp_line):
                            try:
                                temp_dh.data[-1].append(temp_dh.dictionary[word])
                            except:
                                temp_dh.data[-1].append(temp_dh.dictionary['UNK'])

            with open(self.embedding_file, 'wb') as temp_file:
                pickle.dump(temp_dh, temp_file)
            return temp_dh
        else:
            print('Found %s, skipping generation of embeddings.' % self.embedding_file)
            print('Loading embeddings.')
            temp_dh = None
            with open(self.embedding_file, 'rb') as temp_file:
                temp_dh = pickle.load(temp_file)
            return temp_dh

    def load_model(self):
        print('Loading word2vec model %s.' % self.w2v_file_name_extracted)
        self.model = word2vec.Word2Vec.load_word2vec_format(self.w2v_file_name_extracted, binary=True)
        self.model.init_sims(replace=True)

    def maybe_build_sorted_dict(self):
        if not os.path.exists(self.final_dict_file):
            if self.model == None:
                self.load_model()
            ##print(self.model['I'])
            broken_count = 0
            largest_val = ""
            largest_occurence = 0
            failures = []
            for index, values in enumerate(self.sorted_vals_100k):
                try:
                    self.final_dict[values[0]] = self.model[values[0]]
                except:
                    failures.append(values)
                    if largest_val == "":
                        largest_val = values[0]
                        largest_occurence = values[1]
                    broken_count += 1
            print('Number of broken entries %i' % broken_count)
            print('Most common broken value: %s, at %i occurences' % (largest_val, largest_occurence))
            print(failures[:100])
            print('UNK', self.model['UNK'])
            with open(self.final_dict_file, 'wb') as temp_file:
                pickle.dump(self.final_dict, temp_file)
        else:
            print("Found final dictionary file %s, loading." % self.final_dict_file)
            with open(self.final_dict_file, 'rb') as temp_file:
                self.final_dict = pickle.load(temp_file)

    def maybe_calc_frequencies(self):
        if not os.path.exists(self.sorted_words_file):
            print('Calculating frequencies.')
            temp_filenames = ['clean_en_US.blogs.txt',
                              'clean_en_US.news.txt',
                              'clean_en_US.twitter.txt']
            self.frequency_dict = {}
            for i in temp_filenames:
                self.calc_freqnecies(i)
            print(len(self.frequency_dict))
            sorted_vals = sorted(self.frequency_dict.items(), key=lambda x: x[1], reverse=True)
            self.sorted_vals_100k = sorted_vals[:100000]
            with open(self.sorted_words_file, 'wb') as temp_file:
                ## A list of tuples, (word, number_of_uses)
                pickle.dump(self.sorted_vals_100k, temp_file)
        else:
            print('Sorted values found. Loading %s.' % self.sorted_words_file)
            with open(self.sorted_words_file, 'rb') as temp_file:
                self.sorted_vals_100k = pickle.load(temp_file)
            ##print(self.sorted_vals_100k[20:50])

    def calc_freqnecies(self, filename):
        with open(filename, 'r', encoding='utf8') as temp_file:
            for index, line in enumerate(temp_file):
                temp_line = line.strip().split(' ')
                for key, word in enumerate(temp_line):
                    try:
                        self.frequency_dict[word] += 1
                    except:
                        self.frequency_dict[word] = 1

    def maybe_clean(self):
        temp_filenames = ['en_US.blogs.txt', 'en_US.news.txt', 'en_US.twitter.txt']
        new_filenames = []
        for i in temp_filenames:
            if not os.path.exists('clean_' + i):
                print('Cleaning %s.' % i)
                new_filenames.append(self.clean_data(i))
            else:
                print('Found %s, skipping.' % i)

    def clean_data(self, filename):
        ##print(os.listdir())
        with open(filename, 'r', encoding='utf8') as temp_file:
            with open('clean_' + filename, 'a', encoding='utf8') as temp_out_file:
                for line in temp_file:
                    temp_line = self.symbol_changer(line)
                    ##self.regex_keep_apost.sub()
                    temp_line = self.regex_keep_apost.sub(' ', temp_line)
                    temp_line = temp_line.strip().split(' ')
                    temp_line_2 = []
                    ##print(temp_line)
                    for key, word in enumerate(temp_line):
                        if len(word) > 0:
                            if word[0].isupper() and word[1:].islower() and key != 0:
                                temp_word = word
                            else:
                                temp_word = word.lower()
                            temp_word = temp_word.strip()
                            ##if '10' in tmp_word:
                            ##    print(tmp_word)
                            try:
                                ##temp_line[key] = self.contractions[word]
                                temp_word = self.contractions[temp_word]
                            except:
                                pass
                            if temp_word[-2:] == "'s":
                                temp_word = temp_word[:-2]
                            temp_word = self.punct_regex.sub('', temp_word)
                            if temp_word != "":
                                temp_line_2.append(temp_word)
                    temp_line_3 = []
                    for key, word in enumerate(temp_line_2):
                        if len(word) > 1:
                            temp_line_3.append(word)
                        elif word == 'i':
                            temp_line_3.append('I')
                        elif word == 'a':
                            temp_line_3.append('a')
                    if len(temp_line_3) > 0:
                        temp_out_file.write(' '.join(temp_line_3) + '\n')


    def downloader(self):
        self.maybe_download(self.w2v_url, self.w2v_file_name, self.w2v_size)
        self.maybe_download(self.sk_url, self.sk_file_name, self.sk_size)

    def maybe_download(self, url, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urlretrieve(url + filename, filename)
            print('not found', filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                print(##'filename', filename, statinfo.st_size)
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
            )

    def maybe_extract(self):
        if not os.path.exists(self.w2v_file_name_extracted):
            self.extractor(self.w2v_file_name)
        else:
            print('Found %s, skipping.' % self.w2v_file_name_extracted)
        self.extractor(self.sk_file_name)

    def extractor(self, filename):
        print('Attempting to extract %s.' % filename)
        if filename[-3:] == '.gz':
            temp_filename = filename[:-3]
            self.extract_gzip(filename, temp_filename)
        elif filename[-4:] == '.zip':
            temp_filename = filename[:-4]
            self.extract_zip(filename, temp_filename)

    def extract_gzip(self, filename_in, filename_out):
        with gzip.open(filename_in, 'rb') as file_in:
            with open(filename_out, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)

    def extract_zip(self, filename_in, filename_out):
        with zipfile.ZipFile(filename_in, 'r') as zip_file:
            zip_paths = ['final/en_US/en_US.blogs.txt',
                        'final/en_US/en_US.news.txt',
                        'final/en_US/en_US.twitter.txt']
            new_paths = ['en_US.blogs.txt',
                        'en_US.news.txt',
                        'en_US.twitter.txt']
            for key, item in enumerate(zip_paths):

                if not os.path.exists(new_paths[key]):
                    with zip_file.open(item, 'r') as file_in:
                        with open(new_paths[key], 'wb') as file_out:
                            shutil.copyfileobj(file_in, file_out)
                else:
                    print('Found %s, skipping.' % new_paths[key])
    def symbol_changer(self, temp_text):
        return self.regex_replace_characters.sub(
            lambda m: self.replacement_dict_escaped[re.escape(m.group(0))], temp_text)

    def setup_text_replacement(self):
        self.replacement_dict = {
            '%': ' percent ',
            '£': ' pound ',
            '$': ' dollar ',
            '€': ' euro ',
            '&': ' and ',
            '@': ' at ',
            '+': ' plus ',
            '’': "'"
        }
        self.replacement_dict_escaped = dict((re.escape(k), v) for k, v in self.replacement_dict.items())
        self.regex_replace_characters = re.compile('|'.join(self.replacement_dict_escaped.keys()))
        ##self.regex_keep_apost = re.compile(r"[^\P{P}\']")
        self.regex_keep_apost = re.compile(r"[^\w{w}\']+")

        self.punct_regex = re.compile('[%s]' % re.escape(string.punctuation))
        ##remove = regex.compile(ur'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
        self.contractions = {
            "10": "ten",
            "11": "eleven",
            "12": "twelve",
            "15": "fifteen",
            "20": "twenty",
            "30": "thirty",
            "100": "one hundred",
            "2012": "two thousand and twelve",
            "ain't": "is not",
            "aint": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "didn’t": "did not",
            "didnt": "did not",
            "doesn't": "does not",
            "doesnt": "does not",
            "don't": "do not",
            "don’t": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "I'd": "I would",
            "I'll": "I will",
            "I'm": "I am",
            "I’m": "I am",
            "I've": "I have",
            "I’ve": "I have",
            "isn't": "is not",
            "isnt": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "it’s": "it is",
            "It’s": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "o'clock": "of the clock",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there would",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "wasnt": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "would've": "would have",
            "wouldn't": "would not",
            "y'all": "you all",
            "you'd": "you had",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
            }

if __name__ == '__main__':
    word2vec_pretrained_link = 'https://docs.google.com/uc?export=download&confirm=iu5Z&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'
    swiftkey_prediction_data_link = 'https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip'
    word2vec_filename = 'GoogleNews-vectors-negative300.bin.gz'
    swiftkey_filename = 'Coursera-SwiftKey.zip'
    os.chdir('..')
    os.chdir('Datasets')
    dh = DataBuilder(word2vec_pretrained_link, swiftkey_prediction_data_link,
                    word2vec_filename, swiftkey_filename)
    dh.setup()
