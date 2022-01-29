import os 
import sys
import pandas as pd
import numpy as np
from loguru import logger

from os import listdir
from os.path import isfile, join

import random
import warnings
from joblib import dump
from operator import itemgetter

import re
import yaml
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')

warnings.filterwarnings('ignore')
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


@logger.catch
def load_data(data_folder="dataset/"):
    filepaths = [join(data_folder, f) for f in listdir(data_folder) if (isfile(join(data_folder, f))) and ("json" in f)]

    data = []
    target = []

    for ind, f in enumerate(filepaths):        
        json_data = pd.read_json(f, encoding='latin1').values.tolist()
        print(f'File: {f} - {len(json_data)} rows')
        data.extend([item for sublist in json_data for item in sublist])
        target.extend([ind] * len(json_data))
        
    return data, target


@logger.catch
def stopword_removal(text, stop_words, curated_stop_words=None):
    if curated_stop_words is not None and isinstance(curated_stop_words, list):
        stop_words.update(curated_stop_words)

    text = text.lower()
    token = word_tokenize(text)
    
    return ' '.join([w for w in token if not w in stop_words])


@logger.catch
def text_cleaner(text):
    rules = [
        {r'\n': u' '}, # remove new line character
        {r'\t': u' '}, # remove tab character
        {r'ß': u'ss'}, # replace ä with ae
        {r'ä': u'ae'}, # replace ä with ae
        {r'ü': u'ue'}, # replace ü with ue
        {r'ö': u'oe'}, # replace ö with oe
        {r'oe': u'o'}, # replace oe with o
        {r'ue': u'u'}, # replace ue with u
        {r'ae': u'a'}, # replace ae with a
        {r'xxx': u' '}, # remove xxx word
        {r'[^A-Za-zÀ-ž ]': u' '},  # keep only ASCII + European Chars and whitespace, no digits
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''},  # remove spaces at the beginning
        {r'\b[A-Za-zÀ-ž]\b': u''} # remove single character words
    ]
    
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        
    return text.lower()


@logger.catch
def do_stemming(text, stemmer):
    stemmed_words = [stemmer.stem(word) for word in text]
    
    return ''.join(stemmed_words)


@logger.catch
def preprocess_single_text(text, stop_words=None, curated_stop_words=None, stemming=True, stemmer=None):
    text = text_cleaner(text)
    
    if stop_words is not None:
        text = stopword_removal(text, stop_words, curated_stop_words)
    
    if stemming:
        text = do_stemming(text, stemmer)
        
    return text


@logger.catch
def process_data(data, stemming=True, stopwords_locale='german', curated_stop_words=None):
    stemmer = SnowballStemmer(stopwords_locale)
    stop_words = set(stopwords.words(stopwords_locale))
    if curated_stop_words is None:
        curated_stop_words = ['bitte', 'xxx', 'herr', 'frau']
    
    processed_data = [preprocess_single_text(text, 
                                      stop_words=stop_words, 
                                      curated_stop_words=curated_stop_words, 
                                      stemming=stemming, 
                                      stemmer=stemmer) for text in data]
    
    return processed_data


@logger.catch
def random_train_test_split(processed_data, target):
    test_inds = random.sample(range(len(processed_data)), len(processed_data)//5)
    train_inds = list(set(range(len(processed_data))) - set(test_inds))

    train_x = list(itemgetter(*train_inds)(processed_data))
    test_x = list(itemgetter(*test_inds)(processed_data))
    train_y = list(itemgetter(*train_inds)(target))
    test_y = list(itemgetter(*test_inds)(target))
    
    return train_x, test_x, train_y, test_y


@logger.catch
def load_mapping(mapping_file='dataset/mapping.yaml'):
    with open(mapping_file, 'r') as f:
        mapping = yaml.safe_load(f)

    mapping_dict = {}
    _ = [mapping_dict.update({f"{el[0]}":f"{el[1]}"}) for el in 
         [(el.split('=')[0].split('.')[0].split('_')[1], el.split('=')[1].strip()) for el in mapping]]
    
    return mapping_dict
