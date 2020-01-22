'''
this work was reimplemented for local data with spanish
voice recerded for the speaker identification ,
all right reserved errol.mamani@ucsp.edu.pe
compilation: python test.py
'''

from __future__ import division # para forzar la division double
import os
import pickle as cPickle
import numpy as np

from scipy.io import wavfile
from featureextraction import extract_features
# from hmmlearn import hmm

import warnings
warnings.filterwarnings("ignore")
import time

 # debuger
import logging
logging.basicConfig(level=logging.DEBUG) # para usar el debug y mostra



# carpeta donde la lista de test audios estaran
test_dir = "DataTest/"

# carpeta donde esta la lista de modelos a cargar
modelpath = "speaker_models/"


# cargamos a lista de modelos en el archivo
hmm_files = [os.path.join(modelpath,fname) for fname in
               os.listdir(modelpath) if fname.endswith('.hmm')]

# mostramos la lista cargada
for fnames in hmm_files:
	print "modelo: ",fnames

# cargamos el contenido d los modelos con la ruta
models = [cPickle.load(open(fname, 'rb')) for fname in hmm_files]

# cargamos el nombre de los modelos con la ruta
speaker_names = [ fname.split('/')[1].split('.hmm')[0] for fname in hmm_files]
# mostramo nombre de los modelos
logging.debug( speaker_names)

error = 0
total_sample = 0

print("tipear 0 para cargando la lista de archivos para el test y la Prediccion :")

take = int(input())

if take == 0:
	print(take)
	test_file = "lista_audios_test.txt"
	file_paths = open(test_file, 'r')
	for path in file_paths:
		total_sample = total_sample + 1
		path = path.strip()
		# cargamos audio y taza de muestreo x cada uno
		# print "Audio Test: ",test_dir+path
		sound_file = test_dir+path
		sample_rate,audio = wavfile.read(test_dir + path)
		vector = extract_features(audio, sample_rate)
		log_likelihood = np.zeros(len(models))
		# print("log_likelihood", log_likelihood, "log_likelihood size: ", len(log_likelihood), "leng models: ", len(models))
        #iteramos atravez de los modelos
		for i in range(len(models)):
			hmm = models[i] # cogemos el modelo i
			scores = np.array(hmm.score(vector))
			log_likelihood[i] = scores.sum()# [9, 2,4,5,5,...N=len(log_likelihood)]
		# print "log_likelihood", log_likelihood
		winner = np.argmax(log_likelihood)
		print "\t detectado como ", speaker_names[winner]
		checker_name = path.split(".")[0]
		print "locutor: ", checker_name , " = ", speaker_names[winner]
		if speaker_names[winner] not in checker_name:
			error = error + 1
		time.sleep(1.0)
	print"error: ", error, "tatal de muestras: ", total_sample
	accuracy = ((total_sample - error)/ total_sample ) *100
	print("El porcentaje de efectividad para el modelo HMM+ MFCC es:",accuracy, "%" )

else:
    print("exit")
