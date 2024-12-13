from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
import nltk
import string


nltk.download('wordnet')

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



def bag_of_words(input_file, output_folder, output_file_name):
    dataset = pd.read_csv(input_file)
    dataset = dataset.dropna(subset=['text'])  
    article_names = dataset.get('title', [f"Article_{i}" for i in range(len(dataset))])
    texts = dataset['text']
    texts = [text.translate(str.maketrans('', '', string.punctuation)) for text in texts]

    
    lemmatizer = WordNetLemmatizer()
    texts = [' '.join([lemmatizer.lemmatize(word) for word in doc.split()]) for doc in texts]
    vectorizer = CountVectorizer(
        lowercase=True,  
        stop_words='english', 
        token_pattern=r'\b[a-zA-Z]+\b'  
    )


    bow_matrix = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vocab)
    bow_df.insert(0, 'article_name', article_names)

    reverse_label_newsgroups = {v: k for k, v in label_newsgroups.items()}

    dataset['article_name'] = dataset['article_name'].map(reverse_label_newsgroups)

    
    os.makedirs(output_folder, exist_ok=True) 
    output_file_path = os.path.join(output_folder, output_file_name)
    bow_df.to_csv(output_file_path, index=False)
    print(f"Bag of Words saved to {output_file_path}")

input_file = "20_newsgroup_train.csv"
output_folder = "saved_folder"
output_file_name = "bag_of_words_train.csv"

bag_of_words(input_file, output_folder, output_file_name)
