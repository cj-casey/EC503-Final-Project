import joblib
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

X_train = pd.read_csv('transformed_data.csv')
Y_train = pd.read_csv('tfidf_train_label.csv')

#setup matrices for XGBoost
d_train = xgb.DMatrix(X_train, label = Y_train)

#define model parameters
params = {
    'max_depth':6,
    'eta':0.3,
    'objective': 'multi:softprob',
    'num_class': 20,
    'eval_metric': 'mlogloss'
}

clf = xgb.train(params, d_train, num_boost_round = 10)
joblib.dump(clf, 'XGB_decision.pkl')

y_train_pred_temp = clf.predict(d_train)
y_train_pred = [np.argmax(prob) for prob in y_train_pred_temp]

training_CCR = accuracy_score(Y_train, y_train_pred)

print("----------XGBoost TRAINING RESULTS----------")
print(f"Training Accuracy: {training_CCR:.4f}")