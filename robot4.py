#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:59:44 2021

@author: tcicchini
"""
import json as js
import pandas as pd
import pytz
import networkx as nx
import plotly.graph_objects as go
import community as com
import numpy as np
import re
from wordcloud import WordCloud as wc
from gensim.matutils import corpus2csc
from gensim.utils import tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from gensim.models.word2vec import Word2Vec
import nltk
import datetime
import os

def funcion_procesamiento_tweets(nombre_archivo_tweets):
    """
    Esta función recibe como argumento el nombre de archivo donde estén almacenados los tweets y devuelve
    una base de datos procesada
    
    Entrada:
        - nombre archivo de tweets
        - nombre de archivo de salida
    Salida:
        None

    
    """
    # Definimos los campos que vamos a guardar
    data = {
            'tw_id' : [], # id del tweet
            'tw_created_at' : [], # fecha de creacion
            'tw_text' : [], # texto
            'tw_favCount' : [], # cantidad de megustas
            'tw_rtCount' : [], # cantidad de rt
            'tw_qtCount' : [], # cantidad de citas
            'tw_rpCount' : [], # cantidad de comentarios
            'tw_location' : [], # location del tweet (usaremos el place)
            'user_id' : [], # id del usuario
            'user_screenName' : [], # screen name del usuario
            'or_tw_id' : [], # todo lo mismo pero para el tweet original
            'or_tw_created_at' : [],
            'or_tw_text' : [],
            'or_tw_favCount' : [],
            'or_tw_rtCount' : [],
            'or_tw_qtCount' : [],
            'or_tw_rpCount' : [],
            'or_tw_location' : [],
            'or_user_id' : [],
            'or_user_screenName' : [],
            'relacion' : [] # relación entre tweet original y tweet respuesta (RT o QT)
            } 
    
    data_usuario = {
            'id_str' : [],
            'screen_name' : [],
            'description' : [],
            'verified' : [],
            'followers_count' : [],
            'friends_count' : [],
            'listed_count' : [],
            'favourites_count' : [],
            'statuses_count' : [],
            'created_at' : [],
            'location' : []
            }
    # Levantamos el archivo de tweets
    with open(nombre_archivo_tweets, 'r') as f:
        for line in f.readlines():
            tweet = js.loads(line) # Transformo cada línea del archivo de tweets en un json para acceder fácilmente a la meta data
            if 'retweeted_status' in tweet.keys(): # estos serán los relación = RT
                data_usuario['id_str'].append(tweet['user']['id_str'])
                data_usuario['screen_name'].append(tweet['user']['screen_name'])
                try:
                    data_usuario['description'].append(tweet['user']['description'].replace('\n',' '))
                except:
                    data_usuario['description'].append(None)
                data_usuario['verified'].append(tweet['user']['verified'])
                data_usuario['followers_count'].append(tweet['user']['followers_count'])
                data_usuario['friends_count'].append(tweet['user']['friends_count'])
                data_usuario['listed_count'].append(tweet['user']['listed_count'])
                data_usuario['favourites_count'].append(tweet['user']['favourites_count'])
                data_usuario['statuses_count'].append(tweet['user']['statuses_count'])
                data_usuario['created_at'].append(tweet['user']['created_at'])
                data_usuario['location'].append(tweet['user']['location'])
                
                data_usuario['id_str'].append(tweet['retweeted_status']['user']['id_str'])
                data_usuario['screen_name'].append(tweet['retweeted_status']['user']['screen_name'])
                try:
                    data_usuario['description'].append(tweet['retweeted_status']['user']['description'].replace('\n',' '))
                except:
                    data_usuario['description'].append(None)                
                data_usuario['verified'].append(tweet['retweeted_status']['user']['verified'])
                data_usuario['followers_count'].append(tweet['retweeted_status']['user']['followers_count'])
                data_usuario['friends_count'].append(tweet['retweeted_status']['user']['friends_count'])
                data_usuario['listed_count'].append(tweet['retweeted_status']['user']['listed_count'])
                data_usuario['favourites_count'].append(tweet['retweeted_status']['user']['favourites_count'])
                data_usuario['statuses_count'].append(tweet['retweeted_status']['user']['statuses_count'])
                data_usuario['created_at'].append(tweet['retweeted_status']['user']['created_at'])
                data_usuario['location'].append(tweet['retweeted_status']['user']['location'])
                
                data['tw_id'].append(tweet['id_str'])
                data['tw_created_at'].append(tweet['created_at'])
                data['tw_text'].append('None') # el texto del RT es el mismo que el original
                data['tw_favCount'].append(tweet['favorite_count'])
                data['tw_rtCount'].append(tweet['retweet_count'])
                data['tw_qtCount'].append(tweet['quote_count'])
                data['tw_rpCount'].append(tweet['reply_count'])
                try:
                    data['tw_location'].append(tweet['place']['full_name'])
                except:
                    data['tw_location'].append('None')
                data['user_id'].append(tweet['user']['id_str'])
                data['user_screenName'].append(tweet['user']['screen_name'])
                
                data['or_tw_id'].append(tweet['retweeted_status']['id_str'])
                data['or_tw_created_at'].append(tweet['retweeted_status']['created_at'])
                try:
                    data['or_tw_text'].append(tweet['retweeted_status']['extended_tweet']['full_text'].replace('\n',' ')) 
                except:
                    data['or_tw_text'].append(tweet['retweeted_status']['text'].replace('\n',' '))
                data['or_tw_favCount'].append(tweet['retweeted_status']['favorite_count'])
                data['or_tw_rtCount'].append(tweet['retweeted_status']['retweet_count'])
                data['or_tw_qtCount'].append(tweet['retweeted_status']['quote_count'])
                data['or_tw_rpCount'].append(tweet['retweeted_status']['reply_count'])
                try:
                    data['or_tw_location'].append(tweet['retweeted_status']['place']['full_name'])
                except:
                    data['or_tw_location'].append('None')
                data['or_user_id'].append(tweet['retweeted_status']['user']['id_str'])
                data['or_user_screenName'].append(tweet['retweeted_status']['user']['screen_name'])
                
                data['relacion'].append('RT')
            elif 'quoted_status' in tweet.keys():
                
                data_usuario['id_str'].append(tweet['user']['id_str'])
                data_usuario['screen_name'].append(tweet['user']['screen_name'])
                try:
                    data_usuario['description'].append(tweet['user']['description'].replace('\n',' '))
                except:
                    data_usuario['description'].append(None)
                data_usuario['verified'].append(tweet['user']['verified'])
                data_usuario['followers_count'].append(tweet['user']['followers_count'])
                data_usuario['friends_count'].append(tweet['user']['friends_count'])
                data_usuario['listed_count'].append(tweet['user']['listed_count'])
                data_usuario['favourites_count'].append(tweet['user']['favourites_count'])
                data_usuario['statuses_count'].append(tweet['user']['statuses_count'])
                data_usuario['created_at'].append(tweet['user']['created_at'])
                data_usuario['location'].append(tweet['user']['location'])
                
                data_usuario['id_str'].append(tweet['quoted_status']['user']['id_str'])
                data_usuario['screen_name'].append(tweet['quoted_status']['user']['screen_name'])
                try:
                    data_usuario['description'].append(tweet['quoted_status']['user']['description'].replace('\n',' '))
                except:
                    data_usuario['description'].append(None)                
                data_usuario['verified'].append(tweet['quoted_status']['user']['verified'])
                data_usuario['followers_count'].append(tweet['quoted_status']['user']['followers_count'])
                data_usuario['friends_count'].append(tweet['quoted_status']['user']['friends_count'])
                data_usuario['listed_count'].append(tweet['quoted_status']['user']['listed_count'])
                data_usuario['favourites_count'].append(tweet['quoted_status']['user']['favourites_count'])
                data_usuario['statuses_count'].append(tweet['quoted_status']['user']['statuses_count'])
                data_usuario['created_at'].append(tweet['quoted_status']['user']['created_at'])
                data_usuario['location'].append(tweet['quoted_status']['user']['location'])
                
                data['tw_id'].append(tweet['id_str'])
                data['tw_created_at'].append(tweet['created_at'])
                try:
                    data['tw_text'].append(tweet['extended_tweet']['full_text'].replace('\n',' ')) # el texto del RT es el mismo que el original
                except:
                    data['tw_text'].append(tweet['text'].replace('\n',' '))
                data['tw_favCount'].append(tweet['favorite_count'])
                data['tw_rtCount'].append(tweet['retweet_count'])
                data['tw_qtCount'].append(tweet['quote_count'])
                data['tw_rpCount'].append(tweet['reply_count'])
                try:
                    data['tw_location'].append(tweet['place']['full_name'])
                except:
                    data['tw_location'].append('None')
                data['user_id'].append(tweet['user']['id_str'])
                data['user_screenName'].append(tweet['user']['screen_name'])
                
                data['or_tw_id'].append(tweet['quoted_status']['id_str'])
                data['or_tw_created_at'].append(tweet['quoted_status']['created_at'])
                try:
                    data['or_tw_text'].append(tweet['quoted_status']['extended_tweet']['full_text'].replace('\n',' ')) 
                except:
                    data['or_tw_text'].append(tweet['quoted_status']['text'].replace('\n',' '))
                data['or_tw_favCount'].append(tweet['quoted_status']['favorite_count'])
                data['or_tw_rtCount'].append(tweet['quoted_status']['retweet_count'])
                data['or_tw_qtCount'].append(tweet['quoted_status']['quote_count'])
                data['or_tw_rpCount'].append(tweet['quoted_status']['reply_count'])
                try:
                    data['or_tw_location'].append(tweet['quoeted_status']['place']['full_name'])
                except:
                    data['or_tw_location'].append('None')
                data['or_user_id'].append(tweet['quoted_status']['user']['id_str'])
                data['or_user_screenName'].append(tweet['quoted_status']['user']['screen_name'])
                
                data['relacion'].append('QT')                
   
            else:
                pass
            
    data = pd.DataFrame(data)
    data['tw_created_at'] = pd.to_datetime(data['tw_created_at']).apply(lambda x : x.astimezone(pytz.timezone('America/Argentina/Buenos_Aires')))
    data['or_tw_created_at'] = pd.to_datetime(data['or_tw_created_at']).apply(lambda x : x.astimezone(pytz.timezone('America/Argentina/Buenos_Aires')))
    data_usuario = pd.DataFrame(data_usuario).drop_duplicates()
    data_usuario['created_at'] = pd.to_datetime(data_usuario['created_at'])
    return data, data_usuario


def armado_red(interacciones, relacion = ['RT','QT'], CG = True):
    d_enlaces = interacciones[interacciones['relacion'].isin(relacion) == True][['user_screenName',
                                                                                 'or_user_screenName',
                                                                                 'relacion',
                                                                                 ]
                                                                                ].groupby(by = ['user_screenName',
                                                                                                'or_user_screenName']
                                                                                                )['relacion'].count().reset_index()
    d_enlaces.rename({'user_screenName' : 'source',
                      'or_user_screenName' : 'target',
                      'relacion' : 'weight'},
                     axis = 1,
                     inplace = True)                                                                                            
    G = nx.from_pandas_edgelist(d_enlaces,edge_attr = 'weight')
    if CG == True:
        G = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

    comunidades = com.best_partition(G,
                                     weight = 'weight')
    comunidades_to_bowl = dict_to_bowl(comunidades)
    comunidades = {}
    for i, c in enumerate(comunidades_to_bowl):
        for n in c:
            comunidades[n] = i
    nx.set_node_attributes(G,
                           comunidades,
                           'comunidad louvain')
    return G

def diccionario_usuarios_localizacion(data, data_usuario):
    tweets_usuarios_localizados = data[(data.tw_location != 'None')][['user_screenName',
                                                                       'tw_location']
                                                                      ].rename({'user_screenName' : 'screen_name',
                                                                                'tw_location' : 'location'},
                                                                                axis = 1
                                                                                ).append(data[(data.or_tw_location != 'None')][['or_user_screenName',
                                                                                                                                        'or_tw_location']
                                                                                                                                        ].rename({'or_user_screenName' : 'screen_name',
                                                                                                                                                  'or_tw_location' : 'location'},
                                                                                                                                                  axis = 1))
    d_usuarios_localizados = data_usuario.dropna(subset = ['location'])[['screen_name',
                                                                       'location']]
    # Apendeamos ambos datos y nos quedamos, si hay repetición de screen_name, con el primero que aparezca. Esto lo hacemos porque confiamos más en la localización de los tweets que la de los usuarios en sí
    d_usuarios_localizados = tweets_usuarios_localizados.append(d_usuarios_localizados).drop_duplicates(subset = ['screen_name'], keep = 'first')

    # Sacamos tildes y mayúsculas para analizarlos mejor posteriormente
    d_usuarios_localizados.location = d_usuarios_localizados.location.apply(lambda x : x.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').lower().rstrip())
    return d_usuarios_localizados.reset_index(drop = True,)

# ------- Función auxiliar -------
def dict_to_bowl(diccionario):
    """
    Función cuya entrada es un diccionario tipo nodo : comuna.
    
    Sale: lista de listas. Cada sub lista corresponde a una determinada comuna
    """
    item_valor = [[item[1], item[0]] for item in diccionario.items()]
    item_valor = sorted(item_valor)
    set_items = set(diccionario.values())
    lista_bowls = []
    i = 0
    for c in set_items:
        bowl_c = []
        try:
            while item_valor[i][0] == c:
                bowl_c.append(item_valor[i][1]) 
                i += 1
        except:
            pass
        lista_bowls.append(bowl_c)
    return sorted(lista_bowls, key = lambda x: len(x), reverse = True)   

def filtrado_datos_argentinos(data_us_loc, G, porcentaje = 0.5):
    usuarios_argentinos = []
    comunidades = dict_to_bowl(dict(nx.get_node_attributes(G,
                                                           'comunidad louvain')
                                    )
                               )
    str_argentina = 'argentina|ciudad autonoma de buenos aires|buenos aires|cordoba|santa fe|caba|misiones|entre rios|corrientes|chaco|formosa|santiago del estero|salta|jujuy|tucuman|san juan|san luis|la rioja|catamarca|mendoza|la pampa|rio negro|chubut|santa cruz|neuquen|tierra del fuego'
    for c in comunidades:
        if data_us_loc[(data_us_loc.screen_name.isin(c)) & (data_us_loc.location.str.contains(str_argentina) == True)].screen_name.count() > porcentaje * data_us_loc[(data_us_loc.screen_name.isin(c))].screen_name.count():
            usuarios_argentinos.extend(c)

    return usuarios_argentinos   

def trabaja_texto(textos):
    """
    Le pasamos los tweets en una lista, los procesa, aplica modelos, devuelve cosas
    """
    textos = [re.sub(r'http\S+', '', t) for t in textos]
    textos = [list(tokenize(t,
                            lower = True)
                   ) for t in textos]
    stop_es = nltk.corpus.stopwords.words('spanish')
    stop_es.extend(['si','q','k','vos','mas','va','x','h'])
    textos = [list(filter(lambda x : x not in stop_es, t)) for t in textos]
    try:
        textos = list(Phraser(Phrases(textos, min_count = 10, delimiter = ' '))[textos])
    except:
        textos = list(Phraser(Phrases(textos, min_count = 10, delimiter = b' '))[textos])
    modelo_wtv = Word2Vec(min_count=1, # ignora palabras cuya frecuencia es menor a esta
                          window=2, # tamanio de la ventana de contexto
                          vector_size=300, # dimension del embedding
                          sample=6e-5, # umbral para downsamplear palabras muy frecuentes
                          alpha=0.03, # tasa de aprendizaje inicial (entrenamiento de la red neuronal)
                          min_alpha=0.0007, # tasa de aprendizaje minima
                          negative=20)
    modelo_wtv.build_vocab(textos, progress_per = 10000)
    modelo_wtv.train(textos, total_examples = modelo_wtv.corpus_count, epochs=30, report_delay=1)
    modelo_wtv.init_sims(replace=True)

    palabras_clave = ['libertad',
                      'autoritarismo',
                      'muerte',
                      'genocidio',
                      'vacunas',
                      'inflacion',
                      'inflación',
                      'salario',
                      'salarios'
                      ]
    tratamiento_palabras_claves = {}
    for p in palabras_clave:
        try:
            m_sim_w = modelo_wtv.wv.most_similar(positive = p)
            tratamiento_palabras_claves[p] = m_sim_w
        except:
            tratamiento_palabras_claves[p] = 'Esta palabra no está en el corpus'
    base = Dictionary(textos)
    textos = [base.doc2bow(t) for t in textos]
    tfidf_model = TfidfModel(textos,
                             normalize = True)
    textos = [tfidf_model[t] for t in textos]
    m_textos = corpus2csc(textos)
    diccionario_terminos_peso = {base[i] : j[0,0] for i,j in enumerate(m_textos.sum(axis=1))}    
    diccionario_Topicos = {}
    ntop = 6
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

    return tratamiento_palabras_claves, diccionario_terminos_peso, diccionario_Topicos        

def nube_palabras(diccionario_terminos_pesos):  
    nube_palabras = wc(width = 1600,
                       height = 800,
                       max_words = 15,
                       background_color = 'white'
                       ).generate_from_frequencies(diccionario_terminos_pesos)
    return nube_palabras.to_image()

dia_previo = datetime.datetime.today() + datetime.timedelta(days = -1)
fechas_a_trabajar = [f.strftime('_%d_%m_%y') for f in pd.date_range(start = dia_previo + datetime.timedelta(days = -6),
                                                                    end = dia_previo
                                                                    )
                      ]
archivos_a_trabajar = ['campanaNacional{}.txt'.format(f) for f in fechas_a_trabajar] 
path = os.path.abspath(os.path.dirname(__file__))
                                                                                                                          
todo_t = pd.DataFrame()
todo_us = pd.DataFrame()

for i, archivo in enumerate(archivos_a_trabajar):
    try:
        t, us = funcion_procesamiento_tweets(os.path.join(path, archivo))
        todo_t = todo_t.append(t)
        todo_us = todo_us.append(us)
    except:
        pass
todo_t = todo_t[todo_t.or_tw_created_at > dia_previo.astimezone(pytz.timezone('America/Argentina/Buenos_Aires')) + datetime.timedelta(days = -6)]
usuario_loc = diccionario_usuarios_localizacion(todo_t,
                                                todo_us)
G_total = armado_red(todo_t)
usuarios_argentina = filtrado_datos_argentinos(usuario_loc, G_total)
try:
    os.mkdir(os.path.join(path,
                          'Red'))
except:
    pass
path_redes_fecha = os.path.join(path,
                                'Red',
                                'desde{}_hasta{}'.format(fechas_a_trabajar[0],
                                                         fechas_a_trabajar[-1]
                                                         )
                                ) 
os.mkdir(path_redes_fecha
         )
todo_arg = todo_t[(todo_t.user_screenName.isin(usuarios_argentina))|(todo_t.or_user_screenName.isin(usuarios_argentina))]
todo_arg.to_csv(os.path.join(path_redes_fecha,
                           'datos_argentina.csv'
                           ))
G_argentina = armado_red(todo_arg,
                         relacion = ['RT'],
                         CG = False
                         )
nx.write_gexf(G_argentina,
              os.path.join(path_redes_fecha,
                           'red_argentina_RT.gexf'
                           )
              )
ppales_comunas = dict_to_bowl(dict(nx.get_node_attributes(G_argentina,
                                                          'comunidad louvain')
                                   )
                              )[:10]
for i,c in enumerate(ppales_comunas):
    os.mkdir(os.path.join(path_redes_fecha,
                          f'Comuna_{i}'
                          )
             )
    tratamiento_palabras_claves, diccionario_terminos_peso, diccionario_Topicos = trabaja_texto(todo_arg[(todo_arg.user_screenName.isin(c)) | (todo_arg.or_user_screenName.isin(c))].or_tw_text.drop_duplicates())
    
    with open(os.path.join(path_redes_fecha,
                           f'Comuna_{i}',
                           'tratamiento_palabras_claves.txt'
                           ), 'w'
              ) as f:
        f.write(f'{tratamiento_palabras_claves}')
    
    nube_palabras(diccionario_terminos_peso).save(os.path.join(path_redes_fecha,
                                                               f'Comuna_{i}',
                                                               'frecuencia_terminos.png'
                                                               )
                                                  )
    
    for n_top in diccionario_Topicos.keys():
        dist_topicos = diccionario_Topicos[n_top]['distribucion_topicos']
        fig = go.Figure(data = go.Scatterpolar(r = list(dist_topicos.values()),
                                               theta = list(dist_topicos.keys()),
                                               fill = 'toself'
                                               )
                        )
        fig.update_layout(polar = dict(radialaxis = dict(visible = True),
                                       ),
                          showlegend = False)
        fig.write_image(os.path.join(path_redes_fecha,
                                     f'Comuna_{i}',
                                     'distribucion_topicos.png'
                                     )
                        )
        pd.Series(dist_topicos).reset_index().to_csv(os.path.join(path_redes_fecha, f'Comuna_{i}','distribucion_topicos.csv'
                                                                  ),
                                                     index = False
                                                     )
        for top in diccionario_Topicos[n_top]['distribucion_palabras_por_topico']:
            nube_palabras(diccionario_Topicos[n_top]['distribucion_palabras_por_topico'][top]).save(os.path.join(path_redes_fecha,
                                                                                                                 f'Comuna_{i}',
                                                                                                                 f'{top}.png'.replace(' ','_')
                                                                                                                 )
                                                                                                    )
