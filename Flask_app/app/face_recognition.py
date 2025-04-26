import numpy as np
import sklearn
import pickle
import cv2


# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # cascade classifier
model_svm =  pickle.load(open('./model/model_svm.pickle',mode='rb')) # machine learning model (SVM)
pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) # pca dictionary
model_pca = pca_models['pca'] # PCA model
mean_face_arr = pca_models['mean_face'] # Mean Face


