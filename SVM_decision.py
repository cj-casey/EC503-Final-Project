# maybe I,
# i'll fly to your front door some time
# when I can finally get away
# when I can finally get away

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def svm(fs_train_data, train_label, fs_test_data, test_label, gamma = 0.5, c = 1):
    
    clf = SVC(kernel = 'rbf', gamma = gamma, c = c) #surely this doesn't complain. surely
    clf.fit(fs_train_data, train_label)

    y_train_pred = clf.predict(fs_train_data)


    train_ccr = accuracy_score(train_label, y_train_pred) 
    train_f1_score = f1_score(train_label, y_train_pred)
    train_conf_mat = confusion_matrix(train_label, y_train_pred)

    y_test_pred = clf.predict(fs_test_data)

    test_ccr = accuracy_score(test_label, y_test_pred)
    test_f1_score = f1_score(test_label, y_test_pred)
    test_conf_mat = confusion_matrix(test_label, y_test_pred)

    return train_ccr, train_f1_score, train_conf_mat, test_ccr, test_f1_score, test_conf_mat



# #loading training and testing data
# Xtrain_temp_df = pd.read_csv('transformed_data.csv')
# Ytrain_temp_df = pd.read_csv('tfidf_train_label.csv')

# # Xtest_temp_df = pd.read_csv('tfidf_test_data.csv')
# # Ytest_temp_df = pd.read_csv('tfidf_test_label.csv')

# X_train = Xtrain_temp_df.to_numpy()
# Y_train = Ytrain_temp_df.to_numpy()
# Y_train = Y_train.ravel()

# # X_test = Xtest_temp_df.to_numpy()
# # Y_test = Ytest_temp_df.to_numpy()
# # Y_test = Y_test.ravel()


# del Xtrain_temp_df, Ytrain_temp_df 
# # Xtest_temp_df, Ytest_temp_df

# # SVM Time
# clf = SVC(kernel = 'rbf', gamma = 0.5, C = 1.0)
# clf.fit(X_train, Y_train)

# #saving model
# #joblib.dump(clf, 'svm_decision.pkl')

# y_train_pred = clf.predict(X_train)

# # y_test_pred = clf.predict(X_test) 

# CCR_train = accuracy_score(Y_train, y_train_pred)
# # CCR_test = accuracy_score(Y_test, y_test_pred)

# confmat_train = confusion_matrix(Y_train, y_train_pred)
# # confmat_test = confusion_matrix(Y_test, y_test_pred)

# # print("----------SVM TRAINING RESULTS----------")
# # #print(confmat_train)
# # print(f"Training CCR: {CCR_train:.4f}")

# # print("----------TESTING RESULTS----------")
# # print(confmat_test)
# # print(f"Testing CCR: {CCR_test:.4f}")