# Maybe I,
# I'll fly to your front door some time
# When I can finally get away
# When I can finally get away

import joblib 
import numpy as np
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#loading training and testing data
Xtrain_temp_df = pd.read_csv('transformed_data.csv')
Ytrain_temp_df = pd.read_csv('tfidf_train_label.csv')

# Xtest_temp_df = pd.read_csv('tfidf_test_data.csv')
# Ytest_temp_df = pd.read_csv('tfidf_test_label.csv')

X_train = Xtrain_temp_df.to_numpy()
Y_train = Ytrain_temp_df.to_numpy()
Y_train = Y_train.ravel()

# X_test = Xtest_temp_df.to_numpy()
# Y_test = Ytest_temp_df.to_numpy()
# Y_test = Y_test.ravel()


del Xtrain_temp_df, Ytrain_temp_df 
# Xtest_temp_df, Ytest_temp_df

# SVM Time
clf = SVC(kernel = 'rbf', gamma = 0.5, C = 1.0)
clf.fit(X_train, Y_train)

#saving model
joblib.dump(clf, 'svm_decision.pkl')

y_train_pred = clf.predict(X_train)

# y_test_pred = clf.predict(X_test) 

CCR_train = accuracy_score(Y_train, y_train_pred)
# CCR_test = accuracy_score(Y_test, y_test_pred)

confmat_train = confusion_matrix(Y_train, y_train_pred)
# confmat_test = confusion_matrix(Y_test, y_test_pred)

print("----------SVM TRAINING RESULTS----------")
#print(confmat_train)
print(f"Training CCR: {CCR_train:.4f}")

# print("----------TESTING RESULTS----------")
# print(confmat_test)
# print(f"Testing CCR: {CCR_test:.4f}")