'''
this work was reimplemented for local data for spanish casa study
all right reseverd by errol.mamani@ucsp.edu.pe

compilation : python test.py
'''
from __future__ import division # para forzar la division double
import os
import pickle as cPickle
import numpy as np

#import sklearn.mixture.gmm
from scipy.io.wavfile import read
from featureextraction import extract_features
from sklearn.mixture import GaussianMixture as GMM
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

#path donde el audio sera extraido (file audios de prueba)
#source   = "/home/wilderd/Documents/SR/Comparation_Study_SR/gmm_implementation/SampleData/"
source = "dataTest/"

# url donde el modelo training fue guardado (files modelos guardados para el test)
#modelpath = "/home/wilderd/Documents/SR/Comparation_Study_SR/gmm_implementation/Speakers_models/"
modelpath = "Speakers_models/"

gmm_files = [os.path.join(modelpath,fname) for fname in
              os.listdir(modelpath) if fname.endswith('.gmm')]
for fname in gmm_files:
    print(fname)

#print(gmm_files)
#Load the Gaussian gender Models
#modelo = cPickle.load(open('','rb'))

# modelo donde podremos extraer el modelo ya guardado
#modelos = cPickle.load(open('/home/wilderd/Documents/SR/Comparation_Study_SR/gmm_implementation/Speakers_models/Ara.gmm','rb'))


models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]

speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
              in gmm_files]

error = 0
total_sample = 0.0

print("si quieres hacer test a un Audio presiona 1 sino, presiona 0 para completar Audios de Muestra :")
take = int(input().strip())

if (take == 1):
    print("Escriba el nombre del archivo de test:")
    path = input().strip()
    print("ruta: "+source+" nombre:"+path)
 #print("full_path: ",path)
    sr, audio = read(source + path)
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm    = models[i]  # comprobando cada modelo uno por uno
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\t detectado como :", speakers[winner])
    time.sleep(1.0)

elif take == 0:
    #test_file = "testSamplePath.txt"
    test_file = "lista_audios_test.txt"
    file_paths = open(test_file, 'r')
    for path in file_paths:
        total_sample += 1.0
        path = path.strip()
        print("Audio Test: ", path)
        print("url", source+path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i] # comprogra el modelo 1 x 1
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        print("\t detectado como :", speakers[winner])
        checker_name = path.split(".")[0] # sacamos solo el nombre sin .wav
        print("nombre_comprobar: ", checker_name)
        print("locutor: ",checker_name, " = ",speakers[winner])
        if speakers[winner] not in checker_name : # si la prediccion y el test en nombre no considen en nada
            error = error + 1
        time.sleep(1.0)
    print("error : ", error," total muestras: ", total_sample)
    accuracy = ((total_sample - error) / total_sample) * 100
    print("El porcentaje de efectividad (accuracy) de la preuba rendimiento con MFCC + GMM es : ", accuracy, "%")


print("\ el programa se ejecuto correctamente. ")
