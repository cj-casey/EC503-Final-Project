# This is a sample Python script.
import math
import string
from pathlib import Path
import kagglehub
import os
import sklearn.feature_extraction
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy.sparse import coo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import re

label_newsgroups = {
    0: 'alt.atheism.txt',
    1: 'comp.graphics.txt',
    2: 'comp.os.ms-windows.misc.txt',
    3: 'comp.sys.ibm.pc.hardware.txt',
    4: 'comp.sys.mac.hardware.txt',
    5: 'comp.windows.x.txt',
    6: 'misc.forsale.txt',
    7: 'rec.autos.txt',
    8: 'rec.motorcycles.txt',
    9: 'rec.sport.baseball.txt',
    10: 'rec.sport.hockey.txt',
    11: 'sci.crypt.txt',
    12: 'sci.electronics.txt',
    13: 'sci.med.txt',
    14: 'sci.space.txt',
    15: 'soc.religion.christian.txt',
    16: 'talk.politics.guns.txt',
    17: 'talk.politics.mideast.txt',
    18: 'talk.politics.misc.txt',
    19: 'talk.religion.misc.txt'
}

def tfidf(k_fold, min_df = 3, max_df = 0.95, save_csvs=False):
    #tfidf - loads k-th fold of newsgroups dataset and appies TFIDF to it:
        # k_fold - which fold youre selecting
        # min_df - minimum amount of articles required for a word to be included
        # max_df - max frequency of a word across articles that is included
        # save_csvs - set to true to save as CSV
    # load train data
    newsgroups_data = fetch_20newsgroups(subset=all, remove=('headers', 'footers', 'quotes'))
    newsgroups_df = pd.DataFrame([newsgroups_data.data, newsgroups_data.target.tolist()]).T
    newsgroups_df.columns = ['text', 'target']

    targets = pd.DataFrame(newsgroups_data.target_names)
    targets.columns = ['title']
    newsgroups_data = pd.merge(newsgroups_df, targets, left_on='target', right_index=True)

    # k-fold splitting, selecting k_fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = list(skf.split(newsgroups_data['text'],newsgroups_data['target']))[k_fold]

    # load train/test data
    newsgroup_data_train = newsgroups_data.iloc[train_index]
    newsgroup_data_train = newsgroup_data_train.dropna(subset=['text'])
    newsgroup_data_test = newsgroups_data.iloc[test_index]
    newsgroup_data_test = newsgroup_data_test.dropna(subset=['text'])

    # tdif fitting
    tfidf = TfidfVectorizer(stop_words='english', max_df=max_df, min_df=min_df)
    train_result = tfidf.fit_transform(newsgroup_data_train['text']) # fit on train data
    test_result = tfidf.transform(newsgroup_data_test['text']) #transform on test data

    #forming matrix with words as features
    tfidf_train_data = pd.DataFrame(train_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_test_data = pd.DataFrame(test_result.toarray(), columns=tfidf.get_feature_names_out())

    # Get labels
    tfidf_train_label = newsgroup_data_train['target'].reset_index(drop=True)
    tfidf_test_label = newsgroup_data_test['target'].reset_index(drop=True)

    #save all as CSVs
    if(save_csvs):
        tfidf_train_data.to_csv(f"tfidf_train_data_fold{k_fold}.csv")
        tfidf_test_data.to_csv(f"tfidf_test_data_fold{k_fold}.csv")
        # tfidf_train_label.to_csv(f"train_label_fold{k_fold}.csv")
        # tfidf_test_label.to_csv(f"test_label_fold{k_fold}.csv")

    return tfidf_train_data,tfidf_train_label,tfidf_test_data,tfidf_test_label

def tfidf_BoW_input(k_fold, min_df=3, max_df=0.95,save_csvs=False):
    # tfidf - loads k-th fold of BoW data and appies TFIDF to it:
    # k_fold - which fold youre selecting
    # min_df - minimum amount of articles required for a word to be included
    # max_df - max frequency of a word across articles that is included
    # save_csvs - set to true to save as CSV

    # Load train and test data
    train = pd.read_csv(f'BoW_train_data_fold{k_fold}.csv')
    train_label = pd.read_csv(f'BoW_train_label_fold{k_fold}.csv')
    test = pd.read_csv(f'_fold{k_fold}.csv')
    test_label = pd.read_csv(f'BoW_test_label_fold{k_fold}.csv')

    # Calculate document frequency (df) for each word
    word_df = (train > 0).sum(axis=0)  # Counts non-zero entries per column
    num_articles = train.shape[0]

    # Identify columns to drop based on min_df and max_df
    columns_to_drop = train.columns[(word_df < min_df) | (word_df / num_articles > max_df)]
    train.drop(columns=columns_to_drop, inplace=True)
    test.drop(columns=columns_to_drop, inplace=True)

    # Calculate TF-IDF for train and test data
    # Term frequency (TF): divide each element by the row sum
    tf_train = train.div(train.sum(axis=1), axis=0).fillna(0)
    tf_test = test.div(test.sum(axis=1), axis=0).fillna(0)

    # Inverse document frequency (IDF): log(N / df)
    idf = np.log(num_articles / (word_df + 1))  # Adding 1 for smoothing
    idf = idf.loc[train.columns]  # Ensure alignment after dropping columns

    # Compute TF-IDF: element-wise multiplication
    tfidf_train = tf_train * idf
    tfidf_test = tf_test * idf

    # save data as csv
    if(save_csvs):
        tfidf_train.to_csv(f"tfidfBoW_train_data_fold{k_fold}.csv")
        tfidf_test.to_csv(f"tfidfBoW_test_data_fold{k_fold}.csv")

    return tfidf_train,train_label,tfidf_test, test_label
