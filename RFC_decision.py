import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

#loading training and testing data
X_train = pd.read_csv('transformed_data.csv')
Ytrain_temp_df = pd.read_csv('tfidf_train_label.csv')


Y_train = Ytrain_temp_df.to_numpy()
Y_train = Y_train.ravel()

del Ytrain_temp_df  


clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
clf.fit(X_train, Y_train)

joblib.dump(clf,'RFC_decision.pkl')

#Make predictions
y_train_pred = clf.predict(X_train)

training_CCR = accuracy_score(y_train_pred,Y_train)
confmat_train = confusion_matrix(y_train_pred, Y_train)

print("----------Random Forest Classifier TRAINING RESULTS----------")
print(f"Training Accuracy: {training_CCR:.4f}")