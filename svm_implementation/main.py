'''
Speaker recoginiton using SVM
implemented december 2019 by
errol.mamani@ucsp.edu.pe
all right reserved.
'''

from __future__ import print_function
import warnings

import os

from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn import svm
#from  importamos svm de sklearn

import numpy as np

warnings.filterwarnings('ignore')


def extract_mfcc(full_path_audio):
    sample_rate,wave = wavfile.read(full_path_audio)
    # print("sample rate: ", sample_rate)
    mfcc_features = mfcc(wave, samplerate= sample_rate, numcep= 12)
    return mfcc_features

def mfcc_to_fvectors(features):
    # calculamos la media para
    mean = np.mean(features, axis = 0)
    # calculamos tambien la desviacion standard
    std = np.std(features, axis = 0)
    # usamos ambos [mean, std] como feature vectors
    fvec = np.concatenate((mean, std)).tolist()
    return fvec

def buildDataSet(folder):
    fileList = [file for file in os.listdir(folder) if os.path.splitext(file)[1]=='.wav']
    # print("audios: ",len(fileList))
    dataset = {}
    for file in fileList:
        # sacamos el nombre 10491 del archivo
        tmp = file.split('.')[0]
        #print("tmp name: ",tmp)
        full_path = folder+file
        #print("full_path: ",full_path)
        feature = extract_mfcc(full_path)
        fvec = mfcc_to_fvectors(feature)
        dataset[tmp] = []
        #print("name:% ",tmp, fvec)
        # print("names: ", tmp)
        dataset[tmp].append(fvec)

    return dataset

def train_svm(dataset_features):
    #
    model = svm.SVC(kernel = 'rbf', class_weight = "balanced", gamma="auto")
    speaker_names = dataset_features.keys()
    # generate speaker_ids from speaker_names
    spkr_ntoi = {}
    spkr_iton = {}

    i = 0
    for name in speaker_names:
        if name not in spkr_ntoi:
           spkr_ntoi[name] = i
           spkr_iton[i] = name
           i += 1

    speaker_ids = map(lambda n: spkr_ntoi[n], speaker_names)
    print("speaker ids: ",speaker_ids)
    #speaker ids:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # creamos lista de melv_list
    mel_vector = []
    for name in dataset_features.keys():
        trainData = dataset_features[name][0] # para reducir las [[]]
        mel_vector.append(trainData)

    # print("mel 2", mel_vector)
    model.fit(mel_vector,[1,0,2])
    return model



def main():

    print("test")
    train_folder = './training/'
    print(train_folder)
    data_features = buildDataSet(train_folder)
    print("la construccion de data mfcc exitossa")
    modelo_svm = train_svm(data_features)
    print("training exitoso")

    test_folder = './test/'

    test_data = buildDataSet(test_folder)
    cont_score = 0
    for label in test_data.keys():
        feature = test_data[label]
        print("feature: ",feature)
        prediction = modelo_svm.predict(feature)
        print("prediccion: ",prediction)




if __name__ == '__main__' :
    main()
