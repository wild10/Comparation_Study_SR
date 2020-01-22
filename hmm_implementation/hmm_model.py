'''
hmm_implementation using MFCC  for speaker identification
re implemented by errol.mamani@ucsp.edu.pe all right
reserver , november 2019
compilation: python2 hmm_model.py
'''


import os

from python_speech_features import mfcc
from scipy.io import wavfile

from hmmlearn import hmm
from featureextraction import extract_features
import numpy as np
import pickle as cPickle

import warnings
warnings.filterwarnings("ignore")

# debuger
import logging
logging.basicConfig(level=logging.DEBUG) # para usar el debug y mostra

# nombre de la carpeta dataset training
source = "DataSet1/"

# carpeta y archivo donde se guardaran los modelos
dest = "speaker_models/"
train_file = "lista_audios_dataSet.txt"


# leemos la lista de datos del file
file_paths = open(train_file,'r')

# definimos varibles generales del modelo
GMMHMM_Models = {}
states_num = 5
GMM_mix_num = 3
tmp_p = 1.0/(states_num-2)
transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0], \
                              [0, tmp_p, tmp_p, tmp_p , 0], \
                              [0, 0, tmp_p, tmp_p,tmp_p], \
                              [0, 0, 0, 0.5, 0.5], \
                              [0, 0, 0, 0, 1]],dtype=np.float)

startprobPrior = np.array([0.5, 0.5, 0, 0, 0],dtype=np.float)

count = 1 # contador para saber cuantos leeremos
          # para cada persona o locutor
# Extrayendo features para cada locutor (5 wavs x locutor)
features = np.asarray(())
for path in file_paths:
    path = path.strip() # elimina el espaciado enter al final
    # print "path:", path

    # leemos el audio
    sample_rate,audio = wavfile.read(source + path)

    # Extrae 40 dimensiones de MFCC & delta MFCC features
    vector = extract_features(audio,sample_rate)
    print("vector: ",vector.shape)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
    # -> if count == 5: --> edited below
    if count == 5:
        # logging.debug(features)
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        model.fit(features)
        # guardar el modelo entrenado gaussiano
        picklefile = path.split("-")[0]+".hmm"
        print(picklefile)
        cPickle.dump(model,open(dest + picklefile,'wb')) # lectura con wb envez de w
        print( '+ modelo completado para el locutor:',picklefile," con puntos de datos = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
