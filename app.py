from flask import Flask, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity
import math
import threading
from nltk.corpus import wordnet
import time
from multiprocessing import Process, Queue
import multiprocessing
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('punkt')
import csv 
import requests 
import xml.etree.ElementTree as ET 
import os
import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool
import time
import random
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
import random
from nltk.stem.porter import *
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from datetime import datetime
import operator
import sys
###################################################

total_df = pd.read_pickle("Preprocessed_questions_text.pkl")
#preprocessed_text = total_df['preprocessed_text']

import pickle

from tqdm import tqdm
with open('glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


avg_w2v_vectors_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in total_df['non_stopword_removed_preprocessed_text'].values: # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_tr.append(vector)



import flask
app = Flask(__name__)


###################################################

		
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\n", "", phrase)
    return phrase
	
	
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#print ('list of stop words:', stop_words)

def nlp_preprocessing(total_text):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        return string
		

def similarity(docs_arg):
    for i in docs_arg:
        doc_dict[i] = cosine_similarity(train[i],test1)[0][0]
    return doc_dict
        
def cleanpunc(sentence): 
    """function to clean the word of any punctuation or special characters"""
    cleaned = re.sub(r'[?|!|"|#|:|=|+|_|{|}|[|]|-|$|%|^|&|]',r'',str(sentence))
    cleaned = re.sub(r'[.|,|)|(|\|/|-|~|`|>|<|*|$|@|;|→]',r'',cleaned)
    return  cleaned
###################################################



@app.route('/')
def hello_world():
    return flask.render_template('index.html')


@app.route('/crawler')
def index():
    return flask.render_template('crawler.html')

@app.route('/predict', methods=['POST'])
def predict():

    query = request.form.to_dict()
    print(type(query ['review_text']))
    print(query ['review_text'])
    query = cleanpunc(query ['review_text'].lower())
    query = decontracted(query)
    query = nlp_preprocessing(query)
    sentence = query
    start_time = time.time()
    avg_w2v_vectors_cv = []; # the avg-w2v for each sentence/review is stored in this list
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_cv.append(vector)

    #print(len(avg_w2v_vectors_cv))
    #print(len(avg_w2v_vectors_cv[0]))


    #########################################################################################################################################
    doc_dict = dict()
    for i in range(len(avg_w2v_vectors_tr)):
        doc_dict[i] = cosine_similarity([avg_w2v_vectors_tr[i]], avg_w2v_vectors_cv)

    a = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True) [:10]

    ##############################################################################################
    top_items = []

    for i in range(10):
        top_items.append(a[i][0])

    print("--- %s seconds ---" % (time.time() - start_time))

    ################################################################################################
    fw = open('SimilarPosts.txt', 'w')
    for index in top_items:
        #print (total_df.iloc[index,3])
        fw.write(total_df.iloc[index,4] + "\n\n")
        #print("*************************************************************************************************************")
    fw.close()

    with open('SimilarPosts.txt', 'r') as f: 
	    return flask.render_template('similar.html', text=f.read()) 


Tag_Rank = {}

def tag_crawler(url):
    source_code = requests.get(url).text
    soup = BeautifulSoup(source_code, 'html.parser')
    for tag_div in soup.find_all('div', {'class': 'post-taglist'}):
        for tag_link in tag_div.find_all('a'):
            tag = tag_link.string
            if tag in Tag_Rank:
                Tag_Rank[tag] += 1
            else:
                Tag_Rank[tag] = 1

def ques_links_crawler(base_url, end_url, page_limit):
    page_no = 1
    while page_no <= page_limit:
        page_url = base_url + str(page_no) + end_url
        source_code = requests.get(page_url).text
        soup = BeautifulSoup(source_code, 'html.parser')
        if page_no is 1:
            os.system('clear')
        print('crawling page ' + str(page_no) + ': [', end='')
        prev_len = 0
        q_no = 1
        for ques_link in soup.find_all('a', {'class': 'question-hyperlink'}):
            url = 'http://stackoverflow.com/' + ques_link.get('href')
            tag_crawler(url)
            for _ in range(prev_len):
                print('\b', end='')
            print('#', end='')
            p_cent = q_no*2
            percent = '] (' + str(p_cent) + '%)'
            prev_len = len(percent)
            print(percent, end='')
            sys.stdout.flush()
            q_no += 1
        page_no += 1


@app.route('/popular', methods=['POST'])
def pop():

    page = request.form.to_dict()
    print(type(page['pop_text']))
    print(page['pop_text'])
    page_limit = int(page['pop_text'])
    print('starting crawling...')
    ques_links_crawler('http://stackoverflow.com/questions?page=', '&sort=newest', page_limit)
    fw = open('Tags_frequency.txt', 'w')
    for key, value in sorted(Tag_Rank.items(), key=operator.itemgetter(1), reverse=True):
        try:
            fw.write(key + " : " + str(Tag_Rank[key]) + "\n")
        except TypeError:
            continue
    print('\nResult saved to file Tags_frequency.txt')
    fw.close()
    
    with open('Tags_frequency.txt', 'r') as f: 
	    return flask.render_template('populartechstack.html', text=f.read()) 

if __name__ == '__main__':
    app.run()