# A Comparison Study of Speaker Identification models

This project is an implementation and one try to compare three of the traditional Machine learning models
GMM(Gaussian mixtures Model), HMM( Hidden Markov Model) and SVM(Support Vector Machines), although were
implemented most of the models we implement and develop the same input and the same MFCC (as a feature extractor),
adding delta, CMS in and 40 dimensions combined to get a vector of MFCC. on the other hand, SVM is a simple unsupervised
based implementation getting the mean of the MFCC and standard deviation as a list input with their correspondence label.
finally, we use the simple intuitive error and accuracy metric we test in around one hundred audios, recorded with a wav
and our experiment reveals that HMM and GMM are better outperforming in all cases getting around 94 percent accuracy
meanwhile, we also demonstrate that it is possible to implement SVM to identify a small group of persons(as a classifying 2-5)
with high accuracy ~90 percent, but we get totally weak results with more classes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

first of all, this project is developed completed in python with a variation of 3 and 2.7 version due to the compatibility of the
gmm and hmm libraries, we recommend run in a local environment with anaconda and other Prerequisites:

```
hmmlearn
sklearn
numpy
pandas
scipy
cPickle
featureextraction
```

### Installing

most all the package necessary to run use those projects are forward installation using pip
and conda for example:

```
pip install  numpy
```

## Running the tests

the individual projects are divided into training_file.py and test_file.py so you need to train before running the test


```
python test_model.py
```

## Deployment

we use python 2, and python 3 in orderd to efectively use the features extraction lybrary and because the poweful
based of the version, all those compilation are in the top source code of the each file. and we follow the special 
setting of the hmm provided by [hmm] (https://github.com/wblgers/hmm_speech_recognition_demo/blob/master/demo.py).

## Authors

* **wildr10** - *ucsp* - [me](https://github.com/wild10) and
* **milagros**- *UNAMBA*-[mibet] (https://github.com/mibet)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Additional Acknowledgments
* we based on those proyect ,and give the credits for their amazing work
* [hmm] (https://github.com/wblgers/hmm_speech_recognition_demo/blob/master/demo.py)
* [gmm] (https://github.com/abhijeet3922/Speaker-identification-using-GMMs)
* [svm] (https://github.com/narendraj9/speaker-reco)
