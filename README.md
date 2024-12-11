# EC503 Final Project
## By Loren Moreira, Alex Melnick, Bogdan Sadikovic, and Connor Casey
# A Study and Analysis of Different Models and NLP Techniques for Supervised News Classification

## Datasets
|We utilized the 20newsgroup dataset which is available through scikitlearn, and the clickbait dataset which is available here https://huggingface.co/datasets/christinacdl/clickbait_detection_dataset
## Pipeline Architecture
### NLP
Our pipeline supports three NLP feature extraction settings, Bag of Words, TF-IDF, or Both.

Bag of Words is implemented by - Bogdan all you

TF-IDF standalone is implemented using the TFIDF vectorizer from Sci-Kit Learn. By default it utilizes
the values of min_df = 3 and max_df = 0.95. This ensure that words are included 
### Feature Selection
## Chi-Squared
Implemented with scikit-learn. The chi-squared test is used to determine the independence of two events. In this case, the chi-squared test is used to determine the independence of the word features and newsgroup labels. The test utilizes the k hyperparameter, where the features with the k highest Chi-Squared scores are selected. 

## Mutual Information
Implemented with scikit-learn. The mutual information test is used to determine the mutual information between two events. In this case, the mutual information test is used to determine the mutual information between the word features and newsgroup labels. The test utilizes the k hyperparameter, where the features with the k highest mutual information scores are selected.

### Model (SVM, Random Forest, Multinomial Naive Bayes)
## SVM
An RBF-Kernel is used alongside SVM, with a hyperparameter gamma which is set to 0.5 by default. The runtime complexity is denoted to be *O(nSV x d)* where nSV denotes the number of support vectors, and d denotes the dimension of the dataset. Implemented with scikit-learn. 

## Random Forest
Implemented with scikit-learn. Let n_estimators = 100 and random_state = 42 (the meaning of life), with no specifics otherwise for the rest of the hyperparameters. A test with a small subset of the original data did not show alterations in the testing/training CCR for changes in the other hyperparameters that can be set in the black box random forest classifier. 

## Multinomial Naive Bayes
Implemented with scikit-learn. No hyperparameters set. Black box function utilizes Laplace Smoothing to handle missing terms. The algorithm relates the conditional probabilities of events utilizing Bayes’ Theorem. Assumes that features are conditionally independent of each other, chooses class of highest posterior probability, which is chosen by multiplying the prior probability by the probability of the class.


## Guide to Use
### Data Formatting
Can be used with any time of text classification, reading in labelled tabular data.    

The data gets preprocessed by having stop words (commonly used English words) and special character (&, -, etc.) removed, as they can inhibit the ability of the dataset to be analyzed. As such, they are removed by the NLP techniques, before the data gets tabularized.     
Human text data is very unstructured, and incredibly difficult for a model to read and analyze, as there are many nuances that exist within human langauge, especially English. As such, the goal is to create a more structured dataset, which is easier for a model to read and make predictions from. 

### Hyperparameters
Min_DF
Max_DF
K_Top_Features
Gamma
### NLP Technique Selection
Bag of Words (BoW):
The BoW program takes in a dataset to read. It initially pre-processes the data, by removing stop words, and removing any special, non alphabetical characters from the dataset. It then creates a vocabulary of all the words, and then goes through the entire dataset, and checks which words from the vocabulary are being used in each article, and how many times they appear. It then takes all that information, and tabularizes it. The goal is to create sparse, structured data that can be used by other techniques to get more processed data that is easier to read and analyze by other programs. 


TF-IDF:
TF-IDF also pre-processes the data, and removes any stop words or strange characters from its reading. It creates a TF-IDF vectorizer, which does so, along with establishing the maximum DF as 0.95 (Ignores if appears in >95% of words) and the minimum DF as 3(Ignores if <3 words). It then creates a sprace matrix. The goal is to create a sparce matrix that shows the importance of different terms, which allows for better analysis of the data in a structured format. 



### Feature Selections
Both Chi-Squared and Mutual Information are available for feature selection using the FeatureSelection.py class. The class is currently set up for use in pipeline.py with Bag of Words. If you want to run it on TF-IDF, you will need to comment out and uncomment a few lines. Specifically, comment out lines 31, 34, 114, and 116. Uncomment lines 33, 35, 113, and 117. 
