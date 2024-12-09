import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from joblib import parallel_backend

def chi_squared_selection(nlp, features, labels, test_features, test_labels, topk):
    print(f"Feature Selection using Chi-Squared for {nlp}...")

    # Save feature column names
    feature_names = features.columns

    # Convert features to dense if sparse
    if hasattr(features, "toarray"):
        features = features.toarray()
        test_features = test_features.toarray()
    else:
        features = features.to_numpy()
        test_features = test_features.to_numpy()

    # Ensure labels are integers
    if labels.values.dtype != 'int':
        labels, _ = pd.factorize(labels.values.ravel())
        print("Labels converted to integers successfully!")

    if test_labels.values.dtype != 'int':
        test_labels, _ = pd.factorize(test_labels.values.ravel())
        print("Test labels converted to integers successfully!")

    # Flatten labels to 1D
    labels = labels.iloc[:, 1]
    labels = labels.values.ravel()
    test_labels = test_labels.iloc[:, 1]
    test_labels = test_labels.values.ravel()
    print("Labels reshaped to 1D successfully!")

    # Compute Chi-Squared scores
    print("Computing Chi-Squared scores...")
    with parallel_backend('loky', n_jobs=-1):
        chi2_scores, p_values = chi2(features, labels)
    print("Chi-Squared scores computed successfully!")

    # Select the top k features using Chi-Squared
    chi2_selector = SelectKBest(lambda X, y: (chi2_scores, None), k=topk)
    chi2_selected_features = chi2_selector.fit_transform(features, labels)
    chi2_selected_indices = chi2_selector.get_support(indices=True)

    # Apply selection to test data
    test_features_selected = test_features[:, chi2_selected_indices]

    # Map indices to actual feature names
    selected_words = feature_names[chi2_selected_indices]

    # Selected Chi-Squared features
    chi2_feature_scores = pd.DataFrame({
        "Feature": selected_words,
        "Chi2_Score": chi2_scores[chi2_selected_indices],
        "P_Value": p_values[chi2_selected_indices]
    }).sort_values(by="Chi2_Score", ascending=False)
    print("Chi-Squared features selected successfully!")

    # View top 10 features
    print(f"Top 10 of {topk} selected features with Chi-Squared:")
    print(chi2_feature_scores.head(10))

    # Save transformed training data
    transformed_data = pd.DataFrame(chi2_selected_features, columns=selected_words)
    transformed_data.to_csv(f"{nlp}_transformed_train_data_chi_squared.csv", index=False)
    print("Transformed Chi-Squared training dataset saved successfully!")

    # Save transformed test data
    transformed_test_data = pd.DataFrame(test_features_selected, columns=selected_words)
    transformed_test_data.to_csv(f"{nlp}_transformed_test_data_chi_squared.csv", index=False)
    print("Transformed Chi-Squared test dataset saved successfully!")

    print(f"Feature Selection using Chi-Squared for {nlp} completed successfully!")


def mutual_info_selection(nlp, features, labels, test_features, test_labels, topk):
    print(f"Feature Selection using Mutual Information for {nlp}...")

    # Save feature column names
    feature_names = features.columns

    # Convert features to dense if sparse
    if hasattr(features, "toarray"):
        features = features.toarray()
        test_features = test_features.toarray()
    else:
        features = features.to_numpy()
        test_features = test_features.to_numpy()

    # Ensure labels are integers
    if labels.values.dtype != 'int':
        labels, _ = pd.factorize(labels.values.ravel())
        print("Labels converted to integers successfully!")

    if test_labels.values.dtype != 'int':
        test_labels, _ = pd.factorize(test_labels.values.ravel())
        print("Test labels converted to integers successfully!")

    # Flatten labels to 1D
    labels = labels.iloc[:, 1]
    labels = labels.values.ravel()
    test_labels = test_labels.iloc[:, 1]
    test_labels = test_labels.values.ravel()
    print("Labels reshaped to 1D successfully!")

    # Compute Mutual Information scores
    print("Computing Mutual Information scores...")
    with parallel_backend('loky', n_jobs=-1):
        mi_scores = mutual_info_classif(features, labels)
    print("Mutual Information scores computed successfully!")

    # Select the top k features using Mutual Information
    mi_selector = SelectKBest(lambda X, y: (mi_scores, None), k=topk)
    mi_selected_features = mi_selector.fit_transform(features, labels)
    mi_selected_indices = mi_selector.get_support(indices=True)

    # Apply selection to test data
    test_features_selected = test_features[:, mi_selected_indices]

    # Map indices to actual feature names
    selected_words = feature_names[mi_selected_indices]

    # Selected Mutual Information features
    mi_feature_scores = pd.DataFrame({
        "Feature": selected_words,
        "MI_Score": mi_scores[mi_selected_indices]
    }).sort_values(by="MI_Score", ascending=False)
    print("Mutual Information features selected successfully!")

    # View top 10 features
    print(f"Top 10 of {topk} selected features with Mutual Information:")
    print(mi_feature_scores.head(10))

    # Save transformed training data
    transformed_data = pd.DataFrame(mi_selected_features, columns=selected_words)
    transformed_data.to_csv(f"{nlp}_transformed_train_data_mutual_info.csv", index=False)
    print("Transformed Mutual Information training dataset saved successfully!")

    # Save transformed test data
    transformed_test_data = pd.DataFrame(test_features_selected, columns=selected_words)
    transformed_test_data.to_csv(f"{nlp}_transformed_test_data_mutual_info.csv", index=False)
    print("Transformed Mutual Information test dataset saved successfully!")

    print(f"Feature Selection using Mutual Information for {nlp} completed successfully!")


# Adjustments in main
if __name__ == "__main__":
    k = 100
    testrun = False

    train_feature_files = ['tfidf_train_data_fold0.csv']
    train_label_files = ['train_label_fold0.csv']
    test_feature_files = ['tfidf_test_data_fold0.csv']
    test_label_files = ['test_label_fold0.csv']

    for train_feat, train_label, test_feat, test_label in zip(train_feature_files, train_label_files, test_feature_files, test_label_files):
        print(f"Loading data from {train_feat} and {test_feat}...")
        features = pd.read_csv(train_feat, nrows=50 if testrun else None)
        labels = pd.read_csv(train_label, nrows=50 if testrun else None)
        test_features = pd.read_csv(test_feat, nrows=50 if testrun else None)
        test_labels = pd.read_csv(test_label, nrows=50 if testrun else None)
        print("Data loaded successfully!")

        print(f"Features Shape: {features.shape}")
        print(f"Labels Shape: {labels.shape}")
        print(f"Features Type: {type(features)}, Labels Type: {type(labels)}")
        print(f"Labels Unique Values: {np.unique(labels)}")


        chi_squared_selection(train_feat[:-4], features, labels, test_features, test_labels, k)
        mutual_info_selection(train_feat[:-4], features, labels, test_features, test_labels, k)
