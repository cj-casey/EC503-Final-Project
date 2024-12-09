# bouncing off things, 
# and you don't know how you fall 
# your power is drained
# so you can't go through walls

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def nb(fs_train_data, train_label, fs_test_data, test_label):
    
    clf = MultinomialNB()
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