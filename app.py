from flask import Flask, jsonify, request
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import math
import threading
from nltk.corpus import wordnet
import time
from multiprocessing import Process, Queue
import multiprocessing
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('stopwords')

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
import time
import joblib
import pickle
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
filename = 'cluster1.sav'
# load the model from disk
kmeans = joblib.load(filename)
#kmeans = pickle.load(open('cluster2.sav', 'rb'))
topic_data_cluster = pd.read_pickle("topic_data_cluster.pkl")
total_df = pd.read_pickle("Preprocessed_questions_text_no_code.pkl")

doc_dict = dict()

def get_features(texts):
    """Returns use encoding"""
    if type(texts) is str:
        texts = [texts]
        return embed(texts)

from tqdm import tqdm
train = []
#Encoding corpus using USE encoding
for i in tqdm(range(total_df['preprocessed_text_no_code'].shape[0])):
	train.append(get_features(total_df['preprocessed_text_no_code'].values[i]))




# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/
# make sure you have the glove_vectors file
#https://github.com/KashifAS/3-Apply-k-NN-on-Donors-Choose-dataset/blob/master/3_DonorsChoose_KNN-Copy1.ipynb
import pickle

from tqdm import tqdm
with open('glove_vectors', 'rb') as f:
    model = pickle.load(f)
    glove_words =  set(model.keys())


# average Word2Vec
# compute average word2vec for each review.
avg_w2v_vectors_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in total_df['preprocessed_text_no_code'].values: # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in glove_words:
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_vectors_tr.append(vector)

print(len(avg_w2v_vectors_tr))
print(len(avg_w2v_vectors_tr[0]))


# https://www.tutorialspoint.com/flask
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
print ('list of stop words:', stop_words)

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


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	query = request.form.to_dict()
	print(type(query ['review_text']))
	print(query ['review_text'])
	query = cleanpunc(query ['review_text'].lower())
	query = decontracted(query)
	query = nlp_preprocessing(query)
	sentence = query
	avg_w2v_vectors_cv = []; # the avg-w2v for each sentence/review is stored in this list
	vector = np.zeros(300) # as word vectors are of zero length
	cnt_words =0; # num of words with a valid vector in the sentence/review
	for word in sentence.split(): # for each word in a review/sentence
		if word in glove_words:
			vector += model[word]
			cnt_words += 1
	if cnt_words != 0:
		vector /= cnt_words
	avg_w2v_vectors_cv.append(vector)#import time
	test_topic = kmeans.predict(avg_w2v_vectors_cv)[0]
	docs = topic_data_cluster[topic_data_cluster['topic']== test_topic]['id'].values
	#Finding 3 nearest cluster
	cen = dict()
	test = kmeans.transform(avg_w2v_vectors_cv)
	for i in  range(kmeans.n_clusters):
		if i != test_topic:
			cen[i] = test[0][i]
    
	top_items = []
#Sorting which has least distance from each cluster centroid to query point
	a = sorted(cen.items(), key=lambda x: x[1]) [:3]
	
	for i in range(3):
		top_items.append(a[i][0])


	doc1 = topic_data_cluster[topic_data_cluster['topic']== top_items[0]]['id'].values
	doc2 = topic_data_cluster[topic_data_cluster['topic']== top_items[1]]['id'].values
	doc3 = topic_data_cluster[topic_data_cluster['topic']== top_items[2]]['id'].values
	docs = list(docs)
	docs.extend(doc1)
	docs.extend(doc2)
	docs.extend(doc3)
	docs = np.asarray(docs)
	global test1
	test1 = get_features(query)
	start_time = time.time()

######################################################################################################################
	
	n = math.floor(len(docs)/4)
	pool = multiprocessing.Pool(processes=4)
	tasks=[]
# Multiprocessing to compute cosine similarity
	tasks.append(pool.apply_async(similarity,[docs[0:n+1]]))
	tasks.append(pool.apply_async(similarity,[docs[n+1:n+n+1]]))
	tasks.append(pool.apply_async(similarity,[docs[n+n+1:n+n+n+1]]))
	tasks.append(pool.apply_async(similarity,[docs[n+n+n+1:]]))

	pool.close()
	pool.join()
	for each in tasks:
	    each.wait()
	    doc_dict.update(each.get())

######################################################################################################################
	top_items = []
#a[0][0]
	a = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True) [:10]
	print(len(doc_dict))
	for i in range(10):
		top_items.append(a[i][0])
    
	print("--- %s seconds ---" % (time.time() - start_time))
#####################################################################################################################
	text = ""
	i = 0
	for index in top_items:
		text = text + "Doc_Num: " + str(i) + " " + total_df.iloc[index,3] + "<br/>" + "<br/>"
		i = i + 1
    #print("*************************************************************************************************************")
		#print (total_df.iloc[index,4])
		#print("*************************************************************************************************************")

	doc_dict.clear() 
	return text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
