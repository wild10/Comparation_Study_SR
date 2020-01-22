'''
    reimplementation svm model for speaker recognition
    by errol.mamani@ucsp.edu.pe jan,2020

    compilation : python2 main.py

'''
from __future__ import division # para forzar la division double
from svm_model import Reconocedor
import os.path
import time

recognizer = Reconocedor("DataSet2")

def menu():
    global recognizer

    # imprimir menu
    print "\n"
    menu_tabla = """eliga una opcion:
                      1) Entrenar otra vez el svm
                      2) Test
                      3) Exit
               """
    print menu_tabla

    # input del usuario
    key_input = int(raw_input())

    if key_input == 1:
        # entrenar con una nueva data
        recognizer = Reconocedor("DataSet2")

    elif key_input == 2:

        # folder para test
        error = 0
        total_samples = 0
        test_dir = "test_data"
        for speaker_item in os.listdir(test_dir):
            total_samples = total_samples + 1
            # print test
            print "Archivo test: ", speaker_item
            # time.sleep(1.0) # usado para retardar tiempo
            speaker = recognizer.predict("test_data/"+speaker_item)
            print "Detectado como: %s \n" % speaker
            # sacamos nombre para comprobar si es el mismo
            cheker_name = speaker.split("-")[0]
            ## sacamos el error si no coenciden
            if cheker_name not in speaker_item :
                print(cheker_name, "!=", speaker_item)
                error = error + 1
        print "error : ", error, "total de muestras: ", total_samples
        accuracy = ((total_samples - error)/total_samples)*100
        print "El porcentaje de efectividad (accuracy) de la prueba de rendimiento con MFCC + SVM es:", accuracy, "%"



    elif key_input == 3:
        exit(0)

if __name__ == "__main__":
    menu()
