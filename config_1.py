# -*- coding: utf-8 -*-
from pathlib import Path

home = str(Path.home())
output_dir = home + '/aNewHope/refactorInforme/output/'
stopwords = home + '/Ambiente/lista_stopwords.txt'
modelDir =  home + '/aNewHope/refactorInforme/modelos/'

sentiment_dict = {
    '_lbl_POSITIVO':'positivo',
    '_lbl_NEGATIVO':'negativo',
    '_lbl_NEUTRO':'neutro',
    'error':'neutro'
}

stance_dict = {
    '_lbl_aFavor':'a favor',
    '_lbl_enContra':'en contra',
    '_lbl_indecidible':'indecidible',
    'error':'indecidible'
}

criteria_dict = {
    'sentiment':sentiment_dict,
    'stance':stance_dict
}

model_dict = {
    'sentiment': {
        'massa':'{0}modelo_massa1.bin'.format(modelDir),
        'general':'{0}sentiment_general_20191117.joblib'.format(modelDir)
    },
    'stance': {
        'massa':'{0}Massa17Kclean.bin'.format(modelDir),
        'FR':'{0}stance_FR_uncertain.bin'.format(modelDir),
        'CFK': '{0}20190610CFKEqualIndecidibles.bin'.format(modelDir),
        'macri': '{0}20190610MacriEqualIndecidibles.bin'.format(modelDir),
        'kicillof': '{0}20190624KicillofEqualIndecidibles.bin'.format(modelDir),
        'vidal': '{0}20190624VidalEqualIndecidibles.bin'.format(modelDir),
        'albertoBETO': '/home/federico/stancedetection/models/albertoStance/',
        'aysaBETO': '/home/federico/stancedetection/models/AySAStance/',
        'albertoHashtagsPolarizados': '/home/federico/stancedetection/models/AlbertoHashtagsPolarizados/',
        'larreta': '/home/federico/stancedetection/models/larretaStance/',
        'larretaDistant': '/home/federico/stancedetection/models/LarretaDistant/',
        'cfkBETO': '/home/federico/stancedetection/models/CFKBeto/',
        'macriBETO': '/home/federico/stancedetection/models/MacriStance/',
        'massaBETO': '/home/federico/stancedetection/models/MassaBETO/',
        'kiciBETO': '/home/federico/stancedetection/models/KiciBETO/',
        
    }
}

model_tests = {
    'sentiment': {
        'massa': ['{0}modelo_massa1.bin'.format(modelDir)]
    },
    'stance': {
        'massa': ['{0}SM_addcross_67.bin'.format(modelDir), '{0}stance_SM_uncertain.bin'.format(modelDir)],
        'FR': ['{0}stance_FR_uncertain.bin'.format(modelDir)],
        'CFK': ['{0}StanceCFK2500clean.bin'.format(modelDir)],
        'macri': ['{0}StanceMacri2500clean.bin'.format(modelDir)]
    }
}

plot_sentiment_dict = {
    'colores':{
        'POS': '#39E016',
        'NEG': '#EE3629',
        'NEU': '#D9D9CD'
    },
    'color_list': ['NEG', 'POS', 'NEU'],
    'contraste': ['POS', 'NEG'],
    'labels': {'POS' : 'Tweets positivos', 'NEG': 'Tweets negativos'}
}

plot_volumen_dict = {
    'colores':{
        'positivo': '#bab39e',
        'negativo': '#bab39e',
        'neutro': '#bab39e'
    },
    'color_list': ['negativo', 'positivo', 'neutro'],
    'contraste': ['positivo', 'negativo'],
    'labels': {'positivo' : 'Tweets positivos', 'negativo': 'Tweets negativos'}
}

plot_stance_dict = {
    'colores':{
        'a favor': '#39E016',
        'en contra': '#EE3629',
        'indecidible': '#FFFF66'
    },
    'color_list': ['en contra', 'a favor', 'indecidible'],
    'contraste': ['a favor', 'en contra'],
    'labels': {'a favor' : 'Tweets a favor', 'en contra': 'Tweets en contra'}
}

plot_influencers_dict = {
    'colores':{
        'Influencer': '#4daf4a',
        'RT Influencer': '#984ea3',
        'Otros': '#def0e6'
    },
    'color_list': ['Influencer', 'RT Influencer', 'Otros'],
    'contraste': ['Influencer', 'RT Influencer'],
    'labels': {'Influencer' : 'Tweets generados \npor influencers', 'RT Influencer': 'RTs a tweets generados \npor influencers'}
}

plot_composicion_dict = {
    'colores':{
        'Tweet': '#4d9ec9',
        'Retweet': '#594dc9',
        'Reply': '#a04dc9'
    },
    'color_list': ['Tweet', 'Retweet', 'Reply'],
    'contraste': ['Tweet', 'Retweet', 'Reply'],
    'labels': {'Retweet' : 'Retweets', 'Tweet': 'Tweets', 'Reply': 'Replies'}
}

plot_pertenencia_dict = {
    'colores':{
        'dentro': '#4daf4a',
        'fuera': '#984ea3'
    },
    'contraste': ['dentro', 'fuera'],
    'labels': {'dentro' : 'Tweets dentro de la red política', 'fuera' : 'Tweets fuera de la red política'}
}

color_dict = {
    #grafoAr
    'Transversal no polarizado': '#377eb8',
    'Macrismo': '#ff7f00',
    'Internacional en español': '#cbd5e8',
    'Kirchnerismo': '#e41a1c',
    'Litoral': '#4daf4a',
    'Córdoba (y Catamarca)': '#4de14a',
    'Norte': '#4dc34a',
    'Cuyo': '#4da54a',
    'Ruido internacional en otros idiomas': '#cbd5e8',
    'Chubut': '#4d874a',
    'La Pampa': '#4daf4a',
    'La Rioja': '#4daf4a',
    'Randoms 1': '#cbd5e8',
    'Motos Ducati': '#cbd5e8',
    'Randoms Argentina': '#cbd5e8',
    'PortalesInfoARG': '#cbd5e8',
    'Randoms 2': '#cbd5e8',
    'ReformaUniversitaria': '#cbd5e8',
    'Ruidito Las Vegas': '#cbd5e8',
    'Otros': '#cbd5e8',
    'Fuera de la red primaria': '#98a2b5',
    #grafoEcon
    'Exterior: Medios, periodistas y organismos':'#f542f5',
    'Filiación Macrista: PRO y Coalición Civica':'#ff7f00',
    'Espectáculos, Medios, Deportes y Empresas':'#4daf4a',
    'Perspectiva de Mercado y liberalismo económico':'#377eb8',
    'Filiación Kirchnerista-Peronista-Izquierda':'#e41a1c',
    #grafoAr201912
    'Transversal':'#9757ff',
    'Deportes, Medios, Música Transversal': '#850033',
    'Patagonia': '#4daf4a',
    'Santa Fé': '#4daf4a',
    'Misiones': '#4daf4a',
    'Entre Rios': '#4daf4a',
    'Frente Renovador / Peronismo': '#377eb8',
    'Cambiemos': '#ff7f00',
     #grafoAr202012
    'Salta, Tucumán, Norte': '#4daf4a',
    'Cambiemos Light': '#f2ed4b',
    'Cambiemos Hard': '#ffaa54',
    'Internacional': '#cbd5e8',
    'Baja Intensidad Política': '#377eb8',
    'Mendoza': '#4daf4a',
    'Córdoba': '#4def4a',
}

comunidades_ideologicas = ['Macrismo', 'Kirchnerismo', 'Transversal no polarizado', #grafoAr
                           'Transversal', 'Deportes, Medios, Música Transversal',
                           'Frente Renovador / Peronismo', 'Cambiemos', #grafoAr201912
                           'Cambiemos Light', 'Cambiemos Hard', 'Baja Intensidad Política'] #grafoAr202012
