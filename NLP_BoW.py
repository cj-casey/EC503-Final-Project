import torch 
import numpy as np
import matplotlib.pyplot as plt
import string
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter
import sys
import sklearn as sk
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

dataset_location = "NewsGroup_dataset" # Get the dataset file location. --> This can change based on where we up the files.
file_type = ".txt" # Check only .txt files, do not look at .csv file in folder, (It's useless).
saved_folder = "saved_folder"

stop_words = set(stopwords.words('english'))
# File names from folder:
# list_files = ['alt.atheism.txt','comp.graphics.txt','comp.os.ms-windows.misc.txt','alt.atheism.txt',
#               'comp.graphics.txt','comp.os.ms-windows.misc.txt','comp.sys.ibm.pc.hardware.txt',
#               'comp.sys.mac.hardware.txt','comp.windows.x.txt','misc.forsale.txt','rec.autos.txt',
#               'rec.motorcycles.txt','rec.sport.baseball.txt','rec.sport.hockey.txt','sci.crypt.txt'
#                 ,'sci.electronics.txt','sci.med.txt','sci.space.txt','soc.religion.christian.txt',
#               'talk.politics.guns.txt','talk.politics.mideast.txt','talk.politics.misc.txt'
#             ,'talk.religion.misc.txt']



# Get the two csv file names:
list_files = ['20_newsgroup_test.csv', '20_newsgroup_train.csv']


forbidden_words = ['newsgroup', 'from', 'and', 'but', 'is', 'will', 'be'] # IGNORE THIS --> NOT USED RIGHT NOW BUT KEPT JUST IN CASE

file_name = '20_newsgroup_train.csv' 
too_many_appearances = 200 # Use this as a parameter for whether too many words (currently if >200 then remove (This will account per article))

print("Creating bag of words for all articles...\n")


def normalize_line_endings(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().replace(b'\r\n', b'\n')
    with open(file_path, 'wb') as f:
        f.write(content)


def bag_of_words(saved_folder, file_name, too_many_appearances):
    print("Starting Bag Of Words...\n")
    # Ensure the saved_folder exists

    print("Reading the CSV file...")
    dataset = pd.read_csv(file_name)
    dataset = dataset.dropna(subset=['text'])

    # List to store the results
    data = []
    last_article = None

    for index, row in dataset.iterrows():
        text = row['text']
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize the text
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and not word.isdigit()]
        # Remove tokens that are entirely digits and forbidden words
        if forbidden_words:
            tokens = [word for word in tokens if not word.isdigit() and word not in forbidden_words]
        else:
            tokens = [word for word in tokens if not word.isdigit()]
        word_counts = Counter(tokens)
        if too_many_appearances is not None:
            word_counts = {word: count for word, count in word_counts.items() if count <= too_many_appearances}


       
        if 'title' in row and pd.notnull(row['title']):
            article_name = row['title']
        else:
            article_name = f"Article_{index}"


        for word, count in word_counts.items():
            if(last_article != article_name):
                data.append({
                    'Article_Name': article_name,
                    'Word': word,
                    'Num_of_Words': count
                })
                last_article = article_name
            else: 
                data.append({
                    'Article_Name': 0,
                    'Word': word,
                    'Num_of_Words': count
                })

    # Create a DataFrame from the list
    output_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    output_file_name = os.path.splitext(file_name)[0] + '_word_counts.csv'
    output_file_path = os.path.join(saved_folder, output_file_name)
    output_df.to_csv(output_file_path, index=False, encoding='utf-8')

    print(f"Word counts saved to {output_file_path}")

# Since the matrix that is created is full of 0s in the titles to save on space when creating the file, you must run it with this code in order to read it properly:
def read_csv_final(csv_file):
    dataset = pd.read_csv(csv_file)
    last_article = None
    data = []
    # Basically works the same as the code to extract the features, except here, the logic is reversed. 
    for index, row in dataset.iterrows():
        article_name = row['Article_Name']
        word = row['word']
        num_of_words = row['Num_of_Words']
        if(article_name == 0):
            data.append({
                'Article_Name': last_article,
                'Word': word,
                'Num_of_Words': num_of_words
            })
        else:
            data.append({
                'Article_Name': article_name,
                'Word': word,
                'Num_of_Words': num_of_words
            })
            last_article = article_name # If the article name is 0, that means that it is actually the same article as the last one, so it will change when being read. 
            # If the last article is not 0, then it is a new article name --> Thus, the article name will change, as will the last_article variable




for file in list_files:
    bag_of_words(saved_folder, file, too_many_appearances)

# Fun fact: If you try to run this code when you have one of the csv files open, it will crash, due to permission errors, so do not do that. 
sys.exit(0)
