#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:39:30 2021

@author: tcicchini
"""
import os
import time
import pandas as pd
from tweepy import OAuthHandler, Stream, StreamListener

def lector_claves(archivo_claves = 'claves_Twitter_bis.txt'):
    """
    Función auxiliar para levantado de claves
    
    Ingresamos con el nombre del archivo y
    nos devuelve las claves en una lista. Cada elemento corresponde, respectivamente, a:
        CONSUMER_KEY
        CONSUMER_SECRET
        ACCESS_TOKEN
        ACCES_TOKEN_SECRET
    Por default, se define el nombre del archivo de entrada como "claves_Twitter.txt", de forma tal 
    que lo único que hay que hacer es crear ese archivo por única vez con los datos de las claves
    """
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),archivo_claves), 'r') as f:
        claves = f.read().split('\n')
    return claves # Variable de salida, las claves ordenadas

class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list = '', languages = ['es'], follow = None):
        # This handles Twitter authetiminsfication and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(lector_claves()[0], lector_claves()[1])
        auth.set_access_token(lector_claves()[2], lector_claves()[3])
        
        stream = Stream(auth, listener, tweet_mode = 'extended')
        # This line filter Twitter Streams to capture data by the keywords: 
        if len(hash_tag_list) != 0:
            stream.filter(languages = languages,
                          track = hash_tag_list,
                          follow = follow,
                          )
        else:
            stream.sample(languages = languages,)
            
class StdOutListener(StreamListener):
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            t = time.strftime('%d_%m_%y')
            with open(self.fetched_tweets_filename.replace('.txt',f'_{t}.txt'), 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
    def on_error(self, status):
        print(status)

pesca = TwitterStreamer()
filters = ['mauriciomacri',
            'CFKArgentina',
            'alferdez',
            'horaciorlarreta',
            'Kicillofok',
            'PatoBullrich',
            'MaximoKirchner_',
            'WolffWaldo',
            'RandazzoF',
            'libertad',
             'autoritarismo',
             'muerte',
             'genocidio',
             'vacunas',
             'inflacion',
             'inflación',
             'salario',
             'salarios'
             ]
filters.extend(pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),
   			'Candidatos.csv'),
   ).Cuenta.dropna().to_list())            
pesca.stream_tweets(os.path.join(os.path.abspath(os.path.dirname(__file__)),'campanaNacional.txt'),
                    filters
                    )
