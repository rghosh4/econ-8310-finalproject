# -*- coding: utf-8 -*-
"""
Created on Sat May  6 21:24:59 2023

@author: rghosh
"""

import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
pio.renderers.default='browser'

## Updating the Directory and just running the Entire Script should work 
os.chdir("C:/Users/rghosh/Documents/Graduate Curicullum/Spring'23/ECON 8310/Final Project")

# Load the dataset
tattacks = pd.read_csv('pakistanClean.csv')

# Make the entries of the gname column and other text categoricals to lowercase and remove leading/trailing white spaces

tattacks[['gname', 'provstate', 'city','target1']] = tattacks[['gname', 'provstate', 'city','target1']].apply(lambda x: x.str.lower().str.strip())


# Calculate the percentage of attacks for each group
gname_counts = tattacks['gname'].value_counts(normalize=True) * 100

# Get the top 10 groups in descending order
top10_gnames = gname_counts.sort_values(ascending=False).head(10)

# Create a horizontal bar plot 
fig = px.bar(x=top10_gnames.values[::-1],  
             y=top10_gnames.index[::-1],  
             orientation='h')

# Updating the plot layout
fig.update_layout(title='Top 10 Groups (if Identified) by Percentage of Attacks',
                  xaxis_title='Percentage of Attacks',
                  yaxis_title='Group Name',
                  showlegend=False)

# Show the plot
fig.show()


# top 10 group names list
top10_gnames = tattacks['gname'].value_counts().head(10).index.tolist()

# Filter the dataframe for only the top 10 groups
tattacks_top10 = tattacks[tattacks['gname'].isin(top10_gnames)]

# Calculate the percentage of claimed categories within each group
group_counts = tattacks_top10.groupby(['gname', 'claimed']).size().reset_index(name='counts')
group_totals = group_counts.groupby('gname')['counts'].sum().reset_index(name='total')
group_counts = pd.merge(group_counts, group_totals, on='gname')
group_counts['percentage'] = group_counts['counts'] / group_counts['total'] * 100

# tree map 
fig = px.treemap(group_counts,
                 path=['gname', 'claimed'],
                 values='counts',
                 custom_data=['percentage'],
                 title="Top 10 Groups (if Identified) vs Claimed Percentage")

# show percentage in hover
fig.update_traces(hovertemplate='Group: %{label}<br>Claimed: %{parent}<br>Counts: %{value}<br>Percentage: %{customdata[0]:.2f}%')

# Show the plot
fig.show()

# check missing perecentage ( coded with -9 and if non-coded exists)
def missing_percentages(df):
    na_percentages = df.isna().mean().round(4) * 100
    minus9_percentages = (df == -9).mean().round(4) * 100
    summary = pd.DataFrame({'NA (%)': na_percentages, '-9 (%)': minus9_percentages})
    return summary

# Get the summary table of NA and -9 percentages by variable
missing_summary = missing_percentages(tattacks)
# missing_summary.to_csv('missing_summary.csv')




# Create a list of object columns
object_cols = ["region", "provstate", "city", "multiple", "success", "suicide", 
               "attacktype1", "targtype1", "targsubtype1", "corp1", "target1", 
               "claimed", "claimmode", "weaptype1", "weapsubtype1"]

# Create a list of numeric columns
numeric_cols = ["iyear","imonth","iday","latitude","longitude","nkill","nkillus","nkillter","nwound","nwoundus","nwoundte"]



# Filter the DataFrame to keep only the rows where gname is not 'unknown'
tattacks_train = tattacks[tattacks['gname'] != 'unknown']

# Filter the DataFrame to keep only the rows where gname is 'unknown'
tattacks_test = tattacks[tattacks['gname'] == 'unknown']

print(tattacks['gname'].describe())


# Combine top 3 groups and others into a new 'target' column
tattacks_train['target'] = tattacks_train['gname'].apply(lambda x: x if x in ['tehrik-i-taliban pakistan (ttp)', 'baloch republican army (bra)', 'baloch liberation front (blf)'] else 'others')




# Create a list of object columns
object_cols = ["region", "provstate", "city", "multiple", "success", "suicide", 
               "attacktype1", "targtype1", "targsubtype1", "corp1", "target1", 
               "claimed", "claimmode", "weaptype1", "weapsubtype1"]

# Create a list of numeric columns
numeric_cols = ["iyear","imonth","iday","latitude","longitude","nkill","nkillus","nkillter","nwound","nwoundus","nwoundte"]


# Use apply() to convert object columns to categorical data type
tattacks_train[object_cols] = tattacks_train[object_cols].apply(lambda x: x.astype('category'))

X = tattacks_train[object_cols+numeric_cols]
y = tattacks_train['target']

numerical_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()



######### Pipe line ###############
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', MaxAbsScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=True))
])

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])

X=full_processor.fit_transform(X)


# Train a Random Forest Classifier
classifier = RandomForestClassifier()

# Parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': np.arange(50, 550, 50),
    'min_samples_split': np.arange(5,30,5),
    'max_features': ['sqrt', 'log2'],
}

# RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(classifier, param_distributions=param_grid, n_iter=20, cv=5, random_state=42, verbose=2, n_jobs=-1)
random_search.fit(X, y)

# Best parameters found by RandomizedSearchCV
best_params = random_search.best_params_
print("Tuned Random Forest Parameters: ", best_params)

# Best cross-validation score found by RandomizedSearchCV
best_score = random_search.best_score_
print("Best 5-fold Cross Validation Score: {:.2f}%".format(best_score * 100))

#Best 5-fold Cross Validation Score: 67.30%

#################### Prediction of the Unclaimed Attacks  #############################

# Use apply() to convert object columns to categorical data type
tattacks_test[object_cols] = tattacks_test[object_cols].apply(lambda x: x.astype('category'))

X_test = tattacks_test[object_cols+numeric_cols]
X_test =full_processor.transform(X_test)

unknown_preds = random_search.predict(X_test)

prediction_df1 = pd.DataFrame()
prediction_df1['id'] = tattacks_test['eventid']
prediction_df1['t_org'] = unknown_preds

# Calculate the percentage of each predicted class
class_percentages1 = prediction_df1['t_org'].value_counts(normalize=True) * 100

# Convert the series to a DataFrame
prediction_percentage_table1 = class_percentages1.reset_index()

prediction_percentage_table1.columns = ['Class', 'Percentage']

# Display the percentage table
print(prediction_percentage_table1)

#prediction_percentage_table.to_csv("prediction_percentage_table_1.csv")


####################### Feature Importance Plot #################################

# Get feature importances from the best estimator
importances = random_search.best_estimator_.feature_importances_

# Sort importances
indices = np.argsort(importances)
# indices = np.argsort(importances)[-15:]  # Get indices of the top 15 features

# Get the transformed feature names from the column transformer
cat_encoder = full_processor.named_transformers_['category'].named_steps['one-hot']
cat_one_hot_features = cat_encoder.get_feature_names_out(categorical_features)

# Combine the numerical and transformed categorical feature names
transformed_features = np.concatenate([numerical_features, cat_one_hot_features])

# Create a feature importances plot
fig = px.bar(x=importances[indices],
             y=[transformed_features[i] for i in indices],
             orientation='h',
             color=importances[indices])

fig.update_layout(title='Feature Importances',
                  xaxis_title='Relative Importance',
                  yaxis_title='Features')

# Show the plot
# Please Zoom In to see the top Features

fig.show()


###################### Ensemble using a Voting Classifier ###############################
###################### Will Take Significant Time to Run ################################
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Create a function that trains a binary classifier for a specific group
def train_binary_classifier(group_name):
    y_train_binary = (tattacks_train['gname'] == group_name).astype(int)
    classifier = RandomForestClassifier()
    random_search = RandomizedSearchCV(classifier, param_distributions=param_grid, n_iter=20, cv=5, random_state=42, verbose=2, n_jobs=-1)
    random_search.fit(X, y_train_binary)
    return random_search

# Train binary classifiers for each of the top 3 groups
ttp_classifier = train_binary_classifier('tehrik-i-taliban pakistan (ttp)')
bra_classifier = train_binary_classifier('baloch republican army (bra)')
blf_classifier = train_binary_classifier('baloch liberation front (blf)')

# Train a binary classifier for all other groups combined
other_classifier = train_binary_classifier('others')

# Create an ensemble using the VotingClassifier
ensemble_classifier = VotingClassifier(
    estimators=[('ttp', ttp_classifier), ('bra', bra_classifier), ('blf', blf_classifier), ('others', other_classifier)],
    voting='soft')

# Fit the ensemble classifier on the training data
ensemble_classifier.fit(X, y)

# Predict using the ensemble classifier
unknown_preds2 = ensemble_classifier.predict(X_test)


# Perform 5-fold cross-validation on the ensemble classifier
ensemble_cv_scores = cross_val_score(ensemble_classifier, X, y, cv=5)

# Calculate the mean of the cross-validation scores
ensemble_cv_mean = ensemble_cv_scores.mean()

# Print the results
print("Ensemble Mean 5-fold Cross Validation Score: {:.2f}%".format(ensemble_cv_mean * 100))

#Ensemble Mean 5-fold Cross Validation Score: 67.07%

prediction_df2 = pd.DataFrame()
prediction_df2['id'] = tattacks_test['eventid']
prediction_df2['t_org'] = unknown_preds2

class_percentages2 = prediction_df2['t_org'].value_counts(normalize=True) * 100


# Convert the series to a DataFrame
prediction_percentage_table2 = class_percentages2.reset_index()

prediction_percentage_table2.columns = ['Class', 'Percentage']

# Display the percentage table
print(prediction_percentage_table2)

#prediction_percentage_table2.to_csv("prediction_percentage_table_2.csv")



# Merge the two DataFrames on the 'id' column
merged_df = prediction_df1.merge(prediction_df2, on='id', suffixes=('_1', '_2'))

#number of rows where the class predictions are the same
matching_rows = (merged_df['t_org_1'] == merged_df['t_org_2']).sum()

# percentage of matching rows
matching_percentage = (matching_rows / len(merged_df)) * 100

print("Percentage of matching class predictions: {:.2f}%".format(matching_percentage))






