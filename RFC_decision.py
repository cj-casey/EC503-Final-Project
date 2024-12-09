# drunken monologues, confused because
# its not like im falling in love,
# i just want you to do me no good
# and it looks like you could

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def rfc(fs_train_data, train_label, fs_test_data, test_label):
    
    #note, although n is a hyperparemeter, we will let n = 100 for testing. the random state is arbitrary
    # and we will use 42 (the meaning of life) to reproduce results. 

    clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
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