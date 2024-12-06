# Feature Selection using Chi-Square
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from scipy.stats import chi2_contingency

# Options and Hyperparameters
k = 100 # Number of features to select

# Load the features and labels
features = pd.read_csv('tfidf_train_data.csv')
labels = pd.read_csv('tfidf_train_label.csv')

# Ensure labels are in the correct format
labels = labels.values.ravel()

# Make sure number of labels and features match
assert features.shape[0] == labels.shape[0], "Mismatch between features and labels!"

# Compute Chi-Squared scores
chi2_scores, p_values = chi2(features, labels)

# Select the top k features
selector = SelectKBest(chi2, k=k)
selected_features = selector.fit_transform(features, labels)
selected_indices = selector.get_support(indices=True)

# Save the selected top features
feature_scores = pd.DataFrame({
    "Feature": features.columns,
    "Chi2_Score": chi2_scores,
    "P_Value": p_values
}).sort_values(by="Chi2_Score", ascending=False)
feature_scores.to_csv("selected_features.csv", index=False)

# View top 10 features
print(f"Top 10 of {k} selected features:")
print(feature_scores.head(10))  

# Create a new dataframe with only the top k features
selected_feature_names = features.columns[selected_indices]
transformed_data = features[selected_feature_names]

# Save the transformed dataset
transformed_data.to_csv('transformed_data.csv', index=False)