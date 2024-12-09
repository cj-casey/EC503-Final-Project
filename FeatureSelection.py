import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from joblib import parallel_backend

# Options and Hyperparameters
k = 100  # Number of features to select
testrun = False # Set to true if you only want to import the first 50 rows of data (for script testing purposes)

feature_files = ['tfidf_train_data_fold0.csv', ]
label_files = ['train_label_fold0.csv', ]

for feature_file, label_file in zip(feature_files, label_files):
    # Load the features and labels
    print(f"{feature_file} Loading data...")
    if not testrun:
        features = pd.read_csv(feature_file)
        labels = pd.read_csv(label_file)
    else:
        features = pd.read_csv(feature_file, nrows=50)
        labels = pd.read_csv(label_file, nrows=50)
    print("Data loaded successfully!")

    # Save feature column names
    feature_names = features.columns

    # Convert features to dense if sparse
    if hasattr(features, "toarray"):
        features = features.toarray()
    else:
        features = features.to_numpy()

    # Ensure labels are integers
    if labels.values.dtype != 'int':
        labels, _ = pd.factorize(labels.values.ravel())
        print("Labels converted to integers successfully!")

    # Flatten labels to 1D
    labels = labels.values.ravel()  # Convert to NumPy array and flatten
    print("Labels reshaped to 1D successfully!")

    # Compute Chi-Squared scores
    print("Computing Chi-Squared scores...")
    with parallel_backend('loky', n_jobs=-1):  # Use all CPU cores
        chi2_scores, p_values = chi2(features, labels)
    print("Chi-Squared scores computed successfully!")

    # Select the top k features using Chi-Squared
    chi2_selector = SelectKBest(lambda X, y: (chi2_scores, None), k=k)
    chi2_selected_features = chi2_selector.fit_transform(features, labels)
    chi2_selected_indices = chi2_selector.get_support(indices=True)

    # Map indices to actual feature names (words)
    selected_words = feature_names[chi2_selected_indices]

    # Save the selected Chi-Squared features
    chi2_feature_scores = pd.DataFrame({
        "Feature": selected_words,
        "Chi2_Score": chi2_scores[chi2_selected_indices],  # Filter scores for selected features
        "P_Value": p_values[chi2_selected_indices]         # Filter p-values for selected features
    }).sort_values(by="Chi2_Score", ascending=False)
    chi2_feature_scores.to_csv(f"{feature_file[:-4]}_selected_features_chi_squared.csv", index=False)
    print("Chi-Squared features saved successfully!")

    # View top 10 features
    print(f"Top 10 of {k} selected features with Chi-Squared: (6/17)")
    print(chi2_feature_scores.head(10))  

    # Create a new dataframe with only the top k features
    selected_feature_names = feature_names[chi2_selected_indices]
    transformed_data = pd.DataFrame(chi2_selected_features, columns=selected_feature_names)
    print("Transformed Chi-Squared dataset created successfully!")

    # Save the transformed dataset
    transformed_data.to_csv(f"{feature_file[:-4]}_transformed_data_chi_squared.csv", index=False)
    print("Transformed Chi-Squared dataset saved successfully!")


    # Compute Mutual Information scores
    print("Computing Mutual Information scores...")
    with parallel_backend('loky', n_jobs=-1):  # Use all CPU cores
        mi_scores = mutual_info_classif(features, labels)
    print("Mutual Information scores computed successfully!")

    # Select the top k features using Mutual Information
    mi_selector = SelectKBest(lambda X, y: (mi_scores, None), k=k)
    mi_selected_features = mi_selector.fit_transform(features, labels)
    mi_selected_indices = mi_selector.get_support(indices=True)

    # Map indices to actual feature names (words)
    selected_words = feature_names[mi_selected_indices]

    # Save the selected Mutual Information features
    mi_feature_scores = pd.DataFrame({
        "Feature": selected_words,
        "MI_Score": mi_scores[mi_selected_indices]  # Filter scores for selected features
    }).sort_values(by="MI_Score", ascending=False)
    mi_feature_scores.to_csv(f"{feature_file[:-4]}_selected_features_mutual_info.csv", index=False)
    print("Mutual Information features saved successfully!")

    # View top 10  Mutual Information features
    print(f"Top 10 of {k} selected features with Mutual Information:")
    print(mi_feature_scores.head(10))  

    # Create a new dataframe with only the top k features
    selected_feature_names = feature_names[mi_selected_indices]
    transformed_data = pd.DataFrame(mi_selected_features, columns=selected_feature_names)
    print("Transformed Mutual Information dataset created successfully!")

    # Save the transformed dataset
    transformed_data.to_csv(f"{feature_file[:-4]}_transformed_data_mutual_info.csv", index=False)
    print("Transformed Mutual Information dataset saved successfully!")

    print("Feature Selection using Mutual Information completed successfully!")

print("Feature Selection completed successfully!")

