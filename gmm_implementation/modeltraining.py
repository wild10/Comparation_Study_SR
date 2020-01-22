'''
re implemented jaunary  2020 by errol.mamani@ucsp.edu.pe
compilation: python modeltraining.py
'''

import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture as GMM
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

# nombre carpeta dataset
#source   = "development_set/"
source   = "DataSet1/"

# carpeta y archivo donde se guardara los modelos

dest = "Speakers_models/" # carpetas de modelos
train_file = "lista_audios_dataSet.txt"  # archivo de la lista de entrenamiento files.vaw

#dest = "/home/wilderd/Documents/SR/Comparation_Study_SR/gmm_implementation/Speakers_models/"
#train_file = "/home/wilderd/Documents/SR/Comparation_Study_SR/gmm_implementation/development_set_enroll.txt"
file_paths = open(train_file,'r')

count = 1
# Extrayendo features para cada locutor (5 files por locutor)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # leemos el audio
    sample_rate,audio = read(source + path)

    # Extrae 40 dimensiones de MFCC & delta MFCC features
    vector   = extract_features(audio,sample_rate)
    print("vector: ",vector.shape)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # cuando los features son ya 5 para cada locutor se entrena combina y guarda
	# -> if count == 5: --> pasamo abajo
    if count == 5:
        # cambiar max_iter envez de n_iter
        gmm = GMM(n_components =16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)

        # sacando el nombre del modelo para guardar
        picklefile = path.split("-")[0]+".gmm"
        print(picklefile)
        cPickle.dump(gmm,open(dest + picklefile,'wb')) # lectura con wb envez de w
        print( '+ modelo completado para el locutor:',picklefile," con puntos de datos = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1
