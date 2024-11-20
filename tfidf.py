# Output of this script (tidf_test_data.csv, tfidf_test_label.csv, tfidf_train_data.csv, tfidf_train_label.csv) are too large to be uploaded to GitHub. Please run this script to generate them. They will be used in the FeatureSelection.py script. They are automatically ignored by the .gitignore file, so don't worry about needing to delete them.
import string
from pathlib import Path
import kagglehub
import os
import sklearn.feature_extraction
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from scipy.sparse import coo
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

path = kagglehub.dataset_download('crawford/20-newsgroups')
print(path)
list_files = ['altatheism', 'compgraphics', 'composmswindowsmisc', 'altatheism',
              'compgraphics', 'composmswindowsmisc', 'compsysibmpchardware',
              'compsysmachardware', 'compwindowsx', 'miscforsale', 'recautos',
              'recmotorcycles', 'recsportbaseball', 'recsporthockey', 'scicrypt'
    , 'scielectronics', 'scimed', 'scispace', 'socreligionchristian',
              'talkpoliticsguns', 'talkpoliticsmideast', 'talkpoliticsmisc'
    , 'talkreligionmisc']
remove_words = ['subject', 'Subject', 'maxaxaxaxaxaxaxaxaxaxaxaxaxaxax', 'Newsgroup', 'Newsgroup:', 'document_id',
                'Documentid', 'documentid', 'edu', 'com', 'altatheism', 'compgraphics', 'composmswindowsmisc',
                'altatheism',
                'compgraphics', 'composmswindowsmisc', 'compsysibmpchardware',
                'compsysmachardware', 'compwindowsx', 'miscforsale', 'recautos',
                'recmotorcycles', 'recsportbaseball', 'recsporthockey', 'scicrypt'
    , 'scielectronics', 'scimed', 'scispace', 'socreligionchristian',
                'talkpoliticsguns', 'talkpoliticsmideast', 'talkpoliticsmisc'
    , 'talkreligionmisc']
file_path = Path(path)
documents = []
documents_label = []
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


# Courtesy of David Lenz, https://gist.github.com/davidlenz/deff6cc7405d58efa32f4dfe12a6db8b
def twenty_newsgroup_to_csv(train_or_test):
    which_split = 'train'
    if(train_or_test == 1):
        which_split = 'test'
    newsgroups_train = fetch_20newsgroups(subset=which_split, remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame(newsgroups_train.target_names)
    targets.columns = ['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    if (train_or_test == 0):
        out.to_csv('20_newsgroup_train.csv')
    else:
        out.to_csv('20_newsgroup_test.csv')



def tfidf():
    # load train data
    newsgroup_data_train = pd.read_csv('20_newsgroup_train.csv')
    newsgroup_data_train = newsgroup_data_train.dropna(subset=['text'])
    # load test data
    newsgroup_data_test = pd.read_csv('20_newsgroup_test.csv')
    newsgroup_data_test = newsgroup_data_test.dropna(subset=['text'])
    # tdif
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=3);
    train_result = tfidf.fit_transform(newsgroup_data_train['text']) # fit on train data
    test_result = tfidf.transform(newsgroup_data_test['text']) #transform on test data
    # create train data and label
    tfidf_train_data = pd.DataFrame(train_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_train_label = newsgroup_data_train['target']
    #create test data and label
    tfidf_test_data = pd.DataFrame(test_result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_test_label = newsgroup_data_test['target']
    #save all as CSVs
    tfidf_train_data.to_csv('tfidf_train_data.csv', index=False)
    tfidf_train_label.to_csv('tfidf_train_label.csv', index=False)
    tfidf_test_data.to_csv('tfidf_test_data.csv', index=False)
    tfidf_test_label.to_csv('tfidf_test_label.csv', index=False)

    # OPTIONAL print top 10 words by weight

    feature_names = tfidf.get_feature_names_out()
    top_n = 10
    i = 0
    #for doc_idx, doc in enumerate(dense_matrix):
    #    top_indices = np.argsort(doc)[-top_n:][::-1]
    #    label = newsgroup_data['title'].iloc[
    #        doc_idx]
    #    print(f"\nDocument {doc_idx}, Label {label}:")

        # print top n features for articles
    #    for idx in top_indices:
    #        print(f"  {feature_names[idx]}: {doc[idx]}")
    #create final dataset
    #['article #','feature_name1',...'feature_name_n','target','title']
    #tfidf_train_data = pd.DataFrame(dense_matrix,columns = feature_names)
    #tfidf_train_data['label'] = newsgroup_data['target']
    #tfidf_train_data.to_csv('tfidf_train_data.csv', index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tfidf()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
