from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# EC503 Final Project - Code by Connor Casey 12/08/2024
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


def tfidf(k_fold, dataset = '20news', min_df=0.01, max_df=0.9, save_csvs=False, metadata='realistic'):

    # tfidf - loads k-th fold of newsgroups dataset and applies TFIDF to it:
    # k_fold - which fold you're selecting
    # min_df - minimum amount of articles required for a word to be included
    # max_df - max frequency of a word across articles that is included
    # save_csvs - set to true to save as CSV
    # realistic - removes metadata

    # load train data

    if(dataset == '20news'):
        if(metadata == 'realistic'):
            newsgroups_data = fetch_20newsgroups(subset='all',remove=('footers',))
        elif(metadata == 'all'):
            newsgroups_data = fetch_20newsgroups(subset='all')
        else:
            newsgroups_data = fetch_20newsgroups(subset='all',remove=('footers','headers','quotes'))
        # Courtesy of David Lenz, https://gist.github.com/davidlenz/deff6cc7405d58efa32f4dfe12a6db8b
        # he developed a function to read 20newsgroups and process it in a certain way and I integrated it with my code
        # beginning of heavily inspired code
        newsgroups_df = pd.DataFrame([newsgroups_data.data, newsgroups_data.target.tolist()]).T
        newsgroups_df.columns = ['text', 'target']
        targets = pd.DataFrame(newsgroups_data.target_names)
        targets.columns = ['title']
        newsgroups_data = pd.merge(newsgroups_df, targets, left_on='target', right_index=True)
        # end of heavily inspired code
        max_features = 5000
        # ensure target is ints?
        newsgroups_data['target'] = newsgroups_data['target'].astype(int)
    else:
        splits = {'train': 'train.json', 'validation': 'val.json', 'test': 'test.json'}
        newsgroups_data = pd.read_json("hf://datasets/christinacdl/clickbait_detection_dataset/" + splits["train"])
        newsgroups_data.rename(columns={'label':'target'},inplace=True)
        newsgroups_data.drop(columns='text_label', inplace=True)
        max_features = 1500
        # Combine built-in stop words with your custom stop words
    # k-fold splitting, selecting k_fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = list(skf.split(newsgroups_data['text'], newsgroups_data['target']))[k_fold]

    # load train/test data
    newsgroup_data_train = newsgroups_data.iloc[train_index]
    newsgroup_data_train = newsgroup_data_train.dropna(subset=['text'])
    newsgroup_data_test = newsgroups_data.iloc[test_index]
    newsgroup_data_test = newsgroup_data_test.dropna(subset=['text'])

    # tfidf fitting
    # tfidf_model = TfidfVectorizer(stop_words='english',ngram_range =(1,2), max_df=max_df, min_df=min_df,use_idf=True)
    tfidf_model = TfidfVectorizer(stop_words='english',max_features = max_features, min_df = min_df, max_df = max_df)
    train_result = tfidf_model.fit_transform(newsgroup_data_train['text']) # fit on train data
    test_result = tfidf_model.transform(newsgroup_data_test['text']) #transform on test data

    #forming matrix with words as features
    tfidf_train_data = pd.DataFrame(train_result.toarray(), columns=tfidf_model.get_feature_names_out())
    tfidf_test_data = pd.DataFrame(test_result.toarray(), columns=tfidf_model.get_feature_names_out())
    print(tfidf_model.get_feature_names_out())
    # Get labels
    tfidf_train_label = newsgroup_data_train['target'].reset_index(drop=True)
    tfidf_test_label = newsgroup_data_test['target'].reset_index(drop=True)

    #save all as CSVs
    print("Train Data Shape", tfidf_train_data.shape)
    print("Train label shape",tfidf_train_label.shape )
    print("Test Data Shape", tfidf_test_data.shape)
    print("Test label shape", tfidf_test_label.shape)

    if(save_csvs):
        tfidf_train_data.to_csv(f"tfidf_train_data_fold{k_fold}.csv")
        tfidf_test_data.to_csv(f"tfidf_test_data_fold{k_fold}.csv")
        tfidf_train_label.to_csv(f"train_label_fold{k_fold}.csv")
        tfidf_test_label.to_csv(f"test_label_fold{k_fold}.csv")

    return tfidf_train_data,tfidf_train_label,tfidf_test_data,tfidf_test_label

if __name__ == "__main__":
    choice = "1"
    k_fold = 0
    min_df = 0.0
    max_df = 1.0
    save_csvs = False
    if choice == "1":
        tfidf(k_fold,'clickbait',min_df = min_df, max_df = max_df, save_csvs = save_csvs)

    else:
        print("Invalid choice, please select 1 or 2.")