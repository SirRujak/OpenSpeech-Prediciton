#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
from gensim.models import word2vec
import gzip
import shutil
import os
import nltk
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
import embedder

class DataHolder:
    def __init__(self):
        self.dictionary = {}
        self.reverse_dictionary = {}
        self.embedding = []
        self.key = []

class DataBuilder:
    def __init__(self, sk_url, sk_file_name,
                 embedding_filename='SK_full_20k_embedding.pickle'):
        self.sk_url = sk_url
        self.sk_file_name = sk_file_name
        self.sk_file_name_extracted = sk_file_name[:-4]
        self.sk_size = 574661177
        self.stop_words = None
        ## FIXME: Really need to move this to the commented line.
        ## It breaks way too often with just a bad apostrophe.
        self.setup_text_replacement()
        self.model = None
        self.embedding_filename = embedding_filename
        self.word_switch_dict = {}

    def setup(self):
        if not os.path.exists(self.embedding_filename):
            self.downloader()
            self.maybe_extract()
            self.maybe_clean()
        return self.maybe_generate_embedding()

    def maybe_generate_embedding(self):
        if not os.path.exists(self.embedding_filename):
            print('Generating embedding at %s.' % self.embedding_filename)
            temp_obj = embedder.Embedder()
            embeds = temp_obj.build_embedding()
            temp_holder = DataHolder()
            temp_holder.embedding = embeds
            temp_holder.dictionary = temp_obj.dictionary
            temp_holder.reverse_dictionary = temp_obj.reverse_dictionary
            temp_holder.key = temp_obj.count
            with open(self.embedding_filename, 'wb') as temp_file:
                pickle.dump(temp_holder, temp_file)
            print('Done generating embedding.')
            return temp_holder
        else:
            print('Found embedding at %s. Skipping generation. Loading.' % self.embedding_filename)
            with open(self.embedding_filename, 'rb') as temp_file:
                return pickle.load(temp_file)

    def maybe_clean(self):
        temp_filenames = ['en_US.blogs.txt', 'en_US.news.txt', 'en_US.twitter.txt']
        ##new_filenames = []
        for i in temp_filenames:
            if self.stop_words is None:
                try:
                    self.stop_words = set(nltk.corpus.stopwords.words('english'))
                except LookupError:
                    print('Downloading stop word corpus.')
                    nltk.download('stopwords')
                    self.stop_words = set(nltk.corpus.stopwords.words('english'))
            if not os.path.exists('clean_' + i):
                print('Cleaning %s.' % i)
                self.clean_data(i)
                ##new_filenames.append(self.clean_data(i))
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
                        temp_word = word.strip()
                        if len(temp_word) > 0:
                            if temp_word.lower() in self.stop_words:
                                temp_word = word.lower()
                            if len(temp_word) > 1:
                                if temp_word[0].isupper() and temp_word[1:].islower() and key != 0:
                                    temp_word = word
                                else:
                                    temp_word = word.lower()
                            else:
                                temp_word = word.lower()
                            ##if '10' in tmp_word:
                            ##    print(tmp_word)
                            try:
                                temp_word = self.word_switch_dict[temp_word]
                            except:
                                pass
                            if temp_word in self.contractions or temp_word.lower() in self.contractions:
                                try:
                                    ##temp_line[key] = self.contractions[word]
                                    temp_word = self.contractions[temp_word]
                                except:
                                    temp_word = self.contractions[temp_word.lower()]
                            if temp_word[-2:] == "'s":
                                temp_word = temp_word[:-2]
                            temp_word = self.punct_regex.sub('', temp_word)
                            if temp_word != "":
                                temp_line_2.extend(temp_word.split())
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
        if not os.path.exists(self.sk_file_name_extracted):
            self.extractor(self.sk_file_name)
        else:
            print('Found %s, skipping.' % self.sk_file_name_extracted)

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

def generate():
    swiftkey_prediction_data_link = 'https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip'
    embedding_filename = 'SK_full_20k_embedding.pickle'
    swiftkey_filename = 'Coursera-SwiftKey.zip'
    os.chdir('..')
    os.chdir('Datasets')
    data_builder = DataBuilder(swiftkey_prediction_data_link, swiftkey_filename)
    return data_builder.setup()

if __name__ == '__main__':
    generate()
