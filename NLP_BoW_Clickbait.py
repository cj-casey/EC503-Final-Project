import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys

def bag_of_words(input_file, output_folder, output_file_name):
    dataset = pd.read_csv(input_file, delimiter=',', header=0)
    print(f"Columns in dataset: {dataset.columns.tolist()}")
    label_column = 'label' if 'label' in dataset.columns else 'class'
    text_column = 'title' if 'title' in dataset.columns else 'text'

    if label_column not in dataset.columns:
        print(f"Error: '{label_column}' column not found in the input file.")
        return
    if text_column not in dataset.columns:
        print(f"Error: '{text_column}' column not found in the input file.")
        return
    dataset = dataset.dropna(subset=[text_column])


    labels = dataset[label_column]
    texts = dataset[text_column]
    vectorizer = CountVectorizer(
        token_pattern=r'\b[a-zA-Z]+\b'  # Only include alphabetic tokens
    )
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)

    if label_column in bow_df.columns:
        bow_df.drop(columns=[label_column], inplace=True)
    bow_df.insert(0, label_column, labels)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file_name)
    try:
        bow_df.to_csv(output_path, index=False)
        print(f"Bag-of-words matrix saved to {output_path}")
    except PermissionError:
        print(f"Permission denied: Unable to write to {output_path}. Please ensure the file is not open and you have the necessary permissions.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")

if __name__ == "__main__":
    input_file = 'clickbait_data_train.csv'
    output_folder = 'output_folder'
    output_file_name = 'BoW_clickbait_train.csv'
    
    bag_of_words(input_file, output_folder, output_file_name)
    sys.exit(0)
