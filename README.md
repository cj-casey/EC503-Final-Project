# EC503 Final Project
## By Loren Moreira, Alex Melnick, Bogdan Sadikovic, and Connor Casey
# A Study and Analysis of Different Models and NLP Techniques for Supervised News Classification
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
### Hyperparameters
Min_DF
Max_DF
K_Top_Features
gamma
### NLP Technique Selection

### Feature Selections

Both Chi-Squared and Mutual Information are available for feature selection using the FeatureSelection.py class. By running `python pipeline.py`, feature selection using both Chi-Squared and Mutual Information will be performed. Output will be in the terminal and saved to results.txt.
