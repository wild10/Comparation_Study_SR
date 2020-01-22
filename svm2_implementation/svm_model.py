'''
    re implementation based on the speaker-reco
    by errol.mamani@ucsp.edu.pe
    jan 2020
    allright reserved
'''

from __future__ import division

import csv

import os.path

import numpy as np
import scipy.io.wavfile as wavfile

# importamos svm y mfcc
from sklearn import svm
from features import mfcc

# debuger
import logging
logging.basicConfig(level=logging.DEBUG) # para usar el debug y mostrar

class Reconocedor:
    '''
     Esta clase esta implementadad con la finalidad de englobar reconocimiento
     como el entrenamiento, almacenamiento de los datos, test y las funciones necesarias
     para dicho proceso.
    '''

    # constructor de la clase
    def __init__(self , data_dir, out_dir = "aux_folder"):

        self.data_dir = os.path.abspath(data_dir)
        aux_folder = os.path.join(os.getcwd(), "aux_folder")
        saved_file = os.path.join(aux_folder, "training_data.csv")
        #primer intento hacemos el save y luego guardamos
        try:
            os.mkdir(aux_folder) # guardamos en la carpeta
            # creamos el archivo vacio csv par almacenar
            self._generar_features(self.data_dir, saved_file)
        except OSError:
            print("archivos ya creados anteriomente, exception! ")
        # cramos el modelo inicializado
        self.recognizer = svm.SVC()
        melv_list, speaker_names = self._getData(saved_file)
        # generamos speakers_ids, para cada persona
        self.speaker_nameIds = {} # para guardar indices
        self.speaker_idNames = {} # acceso nombres desde ids

        i = 0
        for name in speaker_names:
            if name not in self.speaker_nameIds:
               self.speaker_nameIds[name] = i
               self.speaker_idNames[i] = name
               i += 1

        speaker_ids = map(lambda n: self.speaker_nameIds[n], speaker_names)

        logging.debug(speaker_ids)

        # train a linear svm now
        self.recognizer.fit(melv_list, speaker_ids)

    # funcion para generar la data
    def _generar_features(self, data_dir, outfile):
        '''  generamos el archivo csv conteniendo los labels
             etiquetados  para cada locutor
             '''
        with open(outfile,'w') as abrir:
            melwriter = csv.writer(abrir)
            speakers = os.listdir(data_dir)
            # generar y escribir los features(caraacteristicas) en csv
            for spkr_dir in speakers:
                 # recorremos cada carpeta con sus .wavs
                for soundclip in os.listdir(os.path.join(data_dir, spkr_dir)):
                    # generar los mfcc para cada audio.wav
                    clip_path = os.path.abspath(os.path.join(data_dir, spkr_dir, soundclip))
                    sample_rate, data = wavfile.read(clip_path)
                    ceps = mfcc(data, sample_rate)

                    # sacar los mfcc para cada audio (matriz) y abajo convertimos
                    # a un vector para que luego lo guardemos en un csv(tipo archivo)
                    fvec = self._mfcc_to_fvec(ceps)
                    fvec.append(spkr_dir)

                    # print(fvec) # mostrar los features capturados

                    # guardar(write) una fila en el archivo csv
                    melwriter.writerow(fvec)

    def _mfcc_to_fvec(self, ceps):
        # calcular el promedio
        mean = np.mean(ceps, axis=0)
        # y la desviacion standar de vectores MFCC
        std = np.std(ceps, axis=0)
        # usar [mean, std] como vector de caracteristicas
        fvec = np.concatenate((mean, std)).tolist()

        return fvec


    def _getData(self, icsv):
        """ Devuelve las listas de ejemplo de entrada y salida que se
        enviarán a un clasificador SVM.
        """
        melv_list = []
        speaker_ids = []

        # construimos lista_vector de mesl y speaker_ids
        with open(icsv, 'r') as icsv_handle:
            melreader = csv.reader(icsv_handle) # leer los mels

            for example in melreader:
                melv_list.append(map(float, example[:-1]))
                speaker_ids.append(example[-1])

        # retornar las listas
        return melv_list, speaker_ids

    def predict(self, soundclip):
        '''
           predecimos para un audio a la vez
        '''
        sample_rate, data = wavfile.read(os.path.abspath(soundclip))
        ceps = mfcc(data, sample_rate)
        fvec = self._mfcc_to_fvec(ceps)

        speaker_id = self.recognizer.predict([fvec])[0]
        # print"speaker_id:", speaker_id
        # retornamos los nombres enves de ids
        return self.speaker_idNames[speaker_id]

# if __name__ == "__main__":

#     recognizer = Reconocedor("DataSet")

#     test_dir = os.path.abspath("test_data")
#     testset_size = 0
#     testset_error = 0

#     for spkr_dir in os.listdir(test_dir):
#         for soundclip in os.listdir(os.path.join(test_dir, spkr_dir)):
#             clippath = os.path.abspath(os.path.join(test_dir, spkr_dir, soundclip))
#             print("clippaths:", clippath)
#             prediction = recognizer.predict(clippath)

#             testset_size += 1
#             if prediction != spkr_dir:
#                 testset_error += 1
#                 print "%s %s " % (prediction, u"[\u2717]")
#             else:
#                 print "%s %s " % (prediction, u"[\u2713]")

#     if testset_size == 0:
#         print "No test data available."
#     else:
#         print "Error on test data: %.2f%%\n" % (testset_error / testset_size * 100)
