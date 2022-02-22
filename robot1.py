#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:20:58 2021

@author: tcicchini

La idea de este robot es acceder a los tweets de uno o varios usuarios
de forma periódica.

Además, acceder a los datos de cada usuario.

Finalmente, nos devuelve un informe por cada usuario que contiene:
    - cantidad de seguidores
    - cantidad de...
    - número de tweets publicados
    - a quién retweetea/cita
    
    - palabras más frecuentes
    - tópicos de los tweets
"""

import os
import numpy as np
from tweepy import OAuthHandler, API, Cursor
import datetime
import json
import pandas as pd
import pytz
import plotly.graph_objects as go
import re
from gensim.matutils import corpus2csc
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
import nltk
from wordcloud import WordCloud as wc
import traceback
nltk.download('stopwords')


def lector_claves(archivo_claves = 'claves_Twitter.txt'):
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


def get_tweets_del_usuario(usuarios, count = 200, path_base_datos = ''):
    """
    Función que accede a los últimos count tweets de usuarios, y además, les pide la información de su cuenta.
    
    Genera un archivo .json con los tweets y la data, para cada usuario, poniendo la fecha al final.
    
    """
    auth = OAuthHandler(lector_claves()[0],
                        lector_claves()[1]
                        )
    auth.set_access_token(lector_claves()[2],
                          lector_claves()[3]
                          )
    api = API(auth,
              wait_on_rate_limit = True,
              wait_on_rate_limit_notify = True)
    for usuario in usuarios:
        try:
            os.mkdir(os.path.join(path_base_datos, usuario))
        except:
            pass
        try:
            tweets = [t._json for t in Cursor(api.user_timeline,
                                          screen_name = usuario,
                                          tweet_mode = 'extended',
                                          include_rts = True,
                                          exclude_replies = 'False'
                                          ).items(count)
                  ]
        
            json.dump(tweets,
            	      fp = open(os.path.join(path_base_datos, usuario, f'tweets_{usuario}_{datetime.date.today().strftime("%Y_%m_%d")}.json'
            	                             ),
            	                'w'
            	                )
            	      )
        except:
            pass           

    return None
    
def trabaja_texto(textos):
    """
    Le pasamos los tweets en una lista, los procesa, aplica modelos, devuelve cosas
    """
    textos = [list(tokenize(t,
                            lower = True)
                   ) for t in textos]
    stop_es = nltk.corpus.stopwords.words('spanish')
    textos = [list(filter(lambda x : x not in stop_es, t)) for t in textos]
    try:
        textos = list(Phraser(Phrases(textos, min_count = 2, delimiter = ' '))[textos])
    except:
        textos = list(Phraser(Phrases(textos, min_count = 2, delimiter = b' '))[textos])
    base = Dictionary(textos)
    textos = [base.doc2bow(t) for t in textos]
    if len(textos) > 1:    
        tfidf_model = TfidfModel(textos,
                                 normalize = True)
        textos = [tfidf_model[t] for t in textos]
    m_textos = corpus2csc(textos)
    diccionario_terminos_peso = {base[i] : j[0,0] for i,j in enumerate(m_textos.sum(axis=1))}    
    if len(diccionario_terminos_peso) == 0:
        diccionario_terminos_peso = {'-':1}
    diccionario_Topicos = {}
    if min(m_textos.shape) != 0:
        for ntop in range(2, 7):
            semilla = np.random.seed(1000)
            if ntop <= min(m_textos.shape):
                nmf = NMF(n_components = ntop,
                          random_state = semilla,
                          max_iter = 5000,
                          init = 'nndsvda')
            else:
                nmf = NMF(n_components = ntop,
                          random_state = semilla,
                          max_iter = 5000)        
            norm = Normalizer('l1') # seteamos normalización    
            textos_transformados = nmf.fit_transform(m_textos.T)
            textos_transformados = norm.fit_transform(textos_transformados)
            topicos_terminos = norm.fit_transform(nmf.components_)
            
            diccionario_Topico = {}
            diccionario_Topico['distribucion_topicos'] = {f'Topico {i + 1}' : j for i,
                                                                                    j in enumerate(textos_transformados.sum(axis = 0) / textos_transformados.sum()
                                                                                                   )
                                                          }
            
        
            diccionario_Topico['distribucion_palabras_por_topico'] = {f'Topico {i + 1}' : {base[j] : topicos_terminos[i][j] for j in range(topicos_terminos.shape[1])
                                                                                           } for i in range(topicos_terminos.shape[0])
                                                                      }
            diccionario_Topicos[f'Descomposicion con {ntop} topicos'] = diccionario_Topico
    else:
    	diccionario_Topicos = 'No hay topicos. Un sólo tweet'
    return {'Frecuencia de Terminos' : diccionario_terminos_peso,
            'Topicos' : diccionario_Topicos}

def analisis_tweets_usuario(tweets):
    """
    
    Procesa los tweets, los selecciona por fecha, hace cosas con el texto
    
    """

    # Definimos lo que será el usuario al principio del período y al final del período analizado
    usuario_final = tweets[0]['user']
    
    # Metricas del Usuario
    cantidad_seguidores = usuario_final['followers_count']
    tweets_publicados = len(tweets)
    descripcion_usuario = usuario_final['description']
    cant_favs = sum([t['favorite_count'] for t in tweets])
    cant_rts = sum([t['retweet_count'] for t in tweets])
    # A quiénes comparte
    rt_a = [t['retweeted_status']['user']['screen_name'] for t in tweets if 'retweeted_status' in t.keys()]
    tweets_retweeteados = len(rt_a)
    rt_a = {us : rt_a.count(us) for us in set(rt_a)}
    qt_a = [t['quoted_status']['user']['screen_name'] for t in tweets if 'quoted_status' in t.keys()]
    qt_a = {us : qt_a.count(us) for us in set(qt_a)}
    # Textos (diferenciamos entre originales y retweets)
    
    # - expansión de los textos, sacamos los url
    try:
        textos_or = trabaja_texto([re.sub(r'http\S+', '', t['full_text'])  for t in tweets if 'retweeted_status' not in t.keys()])
    except Exception:
        traceback.print_exc()
        textos_or = 'No hay textos originales'
    try:
        textos_rt = trabaja_texto([re.sub(r'http\S+', '', t['retweeted_status']['full_text']) for t in tweets if 'retweeted_status' in t.keys()])
    except Exception:
        traceback.print_exc()
        textos_rt = 'No hay textos retweeteados'
    
    return {'Cantidad de Seguidores' : cantidad_seguidores,
            'Cantidad de Tweets Total del período' : tweets_publicados,
            'Cantidad de Tweets retweeteados' : tweets_retweeteados,
            'Cantidad de Favs' : cant_favs,
            'Cantidad de Retweets' : cant_rts,
            'Descripcion' : descripcion_usuario,
            'Retweetea a' : rt_a,
            'Cita a' : qt_a,
            'Textos Originales' : textos_or,
            'Textos Replicados' : textos_rt}


def nube_palabras(diccionario_terminos_pesos):  
    nube_palabras = wc(width = 1600,
                       height = 800,
                       max_words = 15,
                       background_color = 'white'
                       ).generate_from_frequencies(diccionario_terminos_pesos)
    return nube_palabras.to_image()
    
def reporte_usuario(usuarios, path_base_datos = '', time_actual = datetime.datetime.today().astimezone(pytz.timezone('America/Argentina/Buenos_Aires'))):

    time_desde = time_actual - datetime.timedelta(days = 7) # Una semana atrás
    for usuario in usuarios:
        try:
            tweets = json.load(open(os.path.join(path_base_datos, usuario, f'tweets_{usuario}_{datetime.date.today().strftime("%Y_%m_%d")}.json'),
                                    'r'
                                    )
                               )
            # Primero, filtramos por fecha
            tweets = [t for t in tweets if pd.to_datetime(t['created_at']).astimezone(pytz.timezone('America/Argentina/Buenos_Aires')) > time_desde]
            # Tenemos toda la info sobre el usuario en formato diccionario
            if len(tweets) != 0:
                analisis_usuario = analisis_tweets_usuario(tweets)
                
                # Armamos un directorio para las fechas analizadas
                dir_usuario_fecha = os.path.join(path_base_datos, usuario, f'desde_{time_desde.strftime("%Y_%m_%d")}_hasta_{time_actual.strftime("%Y_%m_%d")}')
                try:
                    os.mkdir(dir_usuario_fecha)
                except:
                    pass
                pd.DataFrame(data = {'Cantidad de Seguidores' : [analisis_usuario['Cantidad de Seguidores']],
                                     'Cantidad de Tweets Total del período' : [analisis_usuario['Cantidad de Tweets Total del período']],
                                     'Cantidad de Tweets retweeteados' : [analisis_usuario['Cantidad de Tweets retweeteados']],
                                     'Cantidad de Favs' : [analisis_usuario['Cantidad de Favs']],
                                     'Cantidad de Retweets' : [analisis_usuario['Cantidad de Retweets']],
                                     'Descripcion' : [analisis_usuario['Descripcion']]
                                     }
                             ).T.reset_index().to_csv(os.path.join(dir_usuario_fecha,
                                                                   'reporte_metricas.csv'
                                                                   ),
                                                      index = False
                                                      )   
                pd.Series(analisis_usuario['Retweetea a']).sort_values(ascending = False).reset_index().to_csv(os.path.join(dir_usuario_fecha,
                                                                                                                            'reporte_retweets.csv'
                                                                                                                            ),
                                                                                                               index = False
                                                                                                               )       
                pd.Series(analisis_usuario['Cita a']).sort_values(ascending = False).reset_index().to_csv(os.path.join(dir_usuario_fecha,
                                                                                                                       'reporte_citas.csv'
                                                                                                                       ),
                                                                                                          index = False
                                                                                                          )
                if type(analisis_usuario['Textos Originales']) != type(str()):
                    nube_palabras(analisis_usuario['Textos Originales']['Frecuencia de Terminos']).save(os.path.join(dir_usuario_fecha,
                                                                                                                     'terminos_originales.png'
                                                                                                                     )
                                                                                                        )
                    pd.Series(analisis_usuario['Textos Originales']['Frecuencia de Terminos']).reset_index().to_csv(os.path.join(dir_usuario_fecha,
                                                                                                                                 'terminos_originales.csv'
                                                                                                                                 ),
                                                                                                                    index = False
                                                                                                                    )
                    os.mkdir(os.path.join(dir_usuario_fecha, 'Textos_Originales'))
                    if type(analisis_usuario['Textos Originales']['Topicos']) != str:
                        for topico in analisis_usuario['Textos Originales']['Topicos'].keys():
        
                            os.mkdir(os.path.join(dir_usuario_fecha, 'Textos_Originales',topico.replace(' ','_')))
                            distribucion_topicos = analisis_usuario['Textos Originales']['Topicos'][topico]['distribucion_topicos']
                            
                            fig = go.Figure(data = go.Scatterpolar(r = list(distribucion_topicos.values()),
                                                                   theta = list(distribucion_topicos.keys()),
                                                                   fill = 'toself'
                                                                   )
                                            )
                            fig.update_layout(polar = dict(radialaxis = dict(visible = True),
                                                           ),
                                              showlegend = False)
                            fig.write_image(os.path.join(dir_usuario_fecha, 'Textos_Originales',topico.replace(' ','_'), 'distribucion_topicos.png'
                                                        )
                                           )
                            pd.Series(distribucion_topicos).reset_index().to_csv(os.path.join(dir_usuario_fecha, 'Textos_Originales',topico.replace(' ','_'), 'distribucion_topicos.csv'
                                                                                              ),
                                                                                 index = False
                                                                                 )
                            distribucion_palabras_por_topico = analisis_usuario['Textos Originales']['Topicos'][topico]['distribucion_palabras_por_topico']
                            for t, frecuencias in distribucion_palabras_por_topico.items():
                                try:
                                    nube_palabras(frecuencias).save(os.path.join(dir_usuario_fecha, 'Textos_Originales', topico.replace(' ','_'), f'{t.replace(" ","_")}.png'
                                                                                 )
                                                                    )
                                    pd.Series(frecuencias).reset_index().to_csv(os.path.join(dir_usuario_fecha, 'Textos_Originales', topico.replace(' ','_'), f'{t.replace(" ","_")}.csv'
                                                                                             ),
                                                                                index = False
                                                                                )
                                except:
                                    pass
                if type(analisis_usuario['Textos Replicados']) != type(str()):
                    nube_palabras(analisis_usuario['Textos Replicados']['Frecuencia de Terminos']).save(os.path.join(dir_usuario_fecha,
                                                                                                                     'terminos_retweets.png'
                                                                                                                     )
                                                                                                        )
                    pd.Series(analisis_usuario['Textos Replicados']['Frecuencia de Terminos']).reset_index().to_csv(os.path.join(dir_usuario_fecha,
                                                                                                                                 'terminos_retweets.csv'
                                                                                                                                 ),
                                                                                                                    index = False
                                                                                                                    )
                    os.mkdir(os.path.join(dir_usuario_fecha, 'Textos_Replicados'))
                    if type(analisis_usuario['Textos Replicados']['Topicos']) != str:
                        for topico in analisis_usuario['Textos Replicados']['Topicos'].keys():
        
                            os.mkdir(os.path.join(dir_usuario_fecha, 'Textos_Replicados',topico.replace(' ','_')))
                            distribucion_topicos = analisis_usuario['Textos Replicados']['Topicos'][topico]['distribucion_topicos']
        
                            fig = go.Figure(data = go.Scatterpolar(r = list(distribucion_topicos.values()),
                                                                   theta = list(distribucion_topicos.keys()),
                                                                   fill = 'toself'
                                                                   )
                                            )
                            fig.update_layout(polar = dict(radialaxis = dict(visible = True),
                                                           ),
                                              showlegend = False)
                            fig.write_image(os.path.join(dir_usuario_fecha, 'Textos_Replicados', topico.replace(' ','_'), 'distribucion_topicos.png'
                                                        )
                                           )
                            
                            pd.Series(distribucion_topicos).reset_index().to_csv(os.path.join(dir_usuario_fecha, 'Textos_Replicados', topico.replace(' ','_'), 'distribucion_topicos.csv'
                                                                                              ),
                                                                                 index = False
                                                                                 )
                            
                            distribucion_palabras_por_topico = analisis_usuario['Textos Replicados']['Topicos'][topico]['distribucion_palabras_por_topico']
                            for t, frecuencias in distribucion_palabras_por_topico.items():
                                try:
                                    nube_palabras(frecuencias).save(os.path.join(dir_usuario_fecha, 'Textos_Replicados', topico.replace(' ','_'), f'{t.replace(" ","_")}.png'
                                                                                 )
                                                                    )
                                    pd.Series(frecuencias).reset_index().to_csv(os.path.join(dir_usuario_fecha, 'Textos_Replicados', topico.replace(' ','_'), f'{t.replace(" ","_")}.csv'
                                                                                             ),
                                                                                index = False
                                                                                )
                                except:
                                    pass
                json.dump(analisis_usuario,
                          open(os.path.join(dir_usuario_fecha,
                                            'reporte_usuario.json'
                                            ),
                               'w'
                               )
                          )
        except:
            pass
    return None

usuarios = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)),
 				     'Candidatos.csv'),
 			).Cuenta.dropna().to_list()

get_tweets_del_usuario(usuarios,
                        count = 200,
                        path_base_datos = os.path.abspath(os.path.dirname(__file__)))
                       
reporte_usuario(usuarios, path_base_datos = os.path.abspath(os.path.dirname(__file__)))
