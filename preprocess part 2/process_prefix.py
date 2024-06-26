import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, log_loss
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# results_1 = pd.read_csv('results_1.csv')
# results_2 = pd.read_csv('results_2.csv')
# results_3 = pd.read_csv('results_3.csv')
# results_4 = pd.read_csv('results_4.csv')
# results_5 = pd.read_csv('results_5.csv')
# results_6 = pd.read_csv('results_6.csv')

# combine_data = pd.concat([results_1, results_2, results_3,
#                           results_4, results_5, results_6], axis=0)

combine_data = pd.read_csv('old_result/merged_result.csv')

combine_data["status"] = combine_data["semester_3_status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

combine_data = combine_data.reset_index()

# # Scatterplot từng kì
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# sns.scatterplot(x='semester_1_attendance_rate', y='semester_1_average_score',
#                 hue='status', data=combine_data, ax=axs[0])
# sns.scatterplot(x='semester_2_attendance_rate', y='semester_2_average_score',
#                 hue='status', data=combine_data, ax=axs[1])
# sns.scatterplot(x='semester_3_attendance_rate', y='semester_3_average_score',
#                 hue='status', data=combine_data, ax=axs[2])
# plt.show()

# # Histplot Average Score từng kì
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# sns.histplot(data=combine_data, x='semester_1_average_score', hue='status', kde=True, ax=axs[0])
# sns.histplot(data=combine_data, x='semester_2_average_score', hue='status', kde=True, ax=axs[1])
# sns.histplot(data=combine_data, x='semester_3_average_score', hue='status', kde=True, ax=axs[2])
# plt.show()

# # Histplot Attendance Rate từng kì
# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# sns.histplot(data=combine_data, x='semester_1_attendance_rate', hue='status', kde=True, ax=axs[0])
# sns.histplot(data=combine_data, x='semester_2_attendance_rate', hue='status', kde=True, ax=axs[1])
# sns.histplot(data=combine_data, x='semester_3_attendance_rate', hue='status', kde=True, ax=axs[2])
# plt.show()

# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# sns.boxplot(x='status', y='semester_1_attendance_rate', data=combine_data, ax=axs[0])
# sns.boxplot(x='status', y='semester_2_attendance_rate', data=combine_data, ax=axs[1])
# sns.boxplot(x='status', y='semester_3_attendance_rate', data=combine_data, ax=axs[2])
# plt.show()

# fig, axs = plt.subplots(3, 1, figsize=(10, 10))
# sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[0])
# sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[1])
# sns.boxplot(x='status', y='semester_1_average_score', data=combine_data, ax=axs[2])
# plt.show()

decimal_pattern = re.compile(r"Decimal\('(\d+\.\d+)'\)")

combine_data['semester_1'] = combine_data['semester_1'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_2'] = combine_data['semester_2'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))
combine_data['semester_3'] = combine_data['semester_3'].apply(
    lambda x: decimal_pattern.sub(r"'\1'", str(x)))

df_arr = []

# TH xét theo từng kì
df_arr1 = []
df_arr2 = []
df_arr3 = []

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_1'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 1
    df_arr.append(sem_df)
    df_arr1.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_2'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 2
    df_arr.append(sem_df)
    df_arr2.append(sem_df)

for index, row in combine_data.iterrows():
    sem = ast.literal_eval(row['semester_3'])
    sem_df = pd.DataFrame.from_dict(sem, orient='index').reset_index()\
        .rename(columns={'index': 'subject_code'})
    sem_df['student_code'] = row['student_code']
    sem_df['status'] = row['status']
    sem_df['semester'] = 3
    df_arr.append(sem_df)
    df_arr3.append(sem_df)

combine_df = pd.concat(df_arr)
combine_df = combine_df.dropna()
combine_df["attendance_rate"] = combine_df["attendance_rate"].astype(float) / 100
combine_df["average_score"] = combine_df["average_score"].astype(float)
combine_df['total_score'] = combine_df['average_score'] * combine_df['number_of_credit']

combine_1 = pd.concat(df_arr1)
combine_1 = combine_1.dropna()
combine_1['attendance_rate'] = combine_1['attendance_rate'].astype(float) / 100
combine_1['average_score'] = combine_1['average_score'].astype(float)
combine_1_pv = combine_1.pivot(index=['student_code', 'status'], columns='subject_code',
                               values=['attendance_rate', 'average_score'])
combine_1_pv.columns = ['{}_{}'.format(col[0], col[1]) for col in combine_1_pv.columns]
combine_1_pv.reset_index(inplace=True)
combine_1_pv.drop(['attendance_rate_COM3', 'average_score_COM3',
                'attendance_rate_ITI', 'average_score_ITI',
                'attendance_rate_MUL', 'average_score_MUL',
                'attendance_rate_PDP', 'average_score_PDP',
                'attendance_rate_VIE', 'average_score_VIE',
                'attendance_rate_VIE2', 'average_score_VIE2'], axis=1, inplace=True)
for i in combine_1_pv.columns[combine_1_pv.isnull().any(axis=0)]:
    combine_1_pv[i].fillna(combine_1_pv[i].mean(), inplace=True)

combine_2 = pd.concat(df_arr2)
combine_2 = combine_2.dropna()
combine_2['attendance_rate'] = combine_2['attendance_rate'].astype(float) / 100
combine_2['average_score'] = combine_2['average_score'].astype(float)
combine_2_pv = combine_2.pivot(index=['student_code', 'status'], columns='subject_code',
                               values=['attendance_rate', 'average_score'])
combine_2_pv.columns = ['{}_{}'.format(col[0], col[1]) for col in combine_2_pv.columns]
combine_2_pv.reset_index(inplace=True)
combine_2_pv.drop(['attendance_rate_COM4', 'average_score_COM4',
                'attendance_rate_COM5', 'average_score_COM5',
                'attendance_rate_MOB2', 'average_score_MOB2',
                'attendance_rate_MUL', 'average_score_MUL',
                'attendance_rate_MUL2', 'average_score_MUL2',
                'attendance_rate_NET', 'average_score_NET',
                'attendance_rate_SOA', 'average_score_SOA',
                'attendance_rate_VIE2', 'average_score_VIE2'], axis=1, inplace=True)
for i in combine_2_pv.columns[combine_2_pv.isnull().any(axis=0)]:
    combine_2_pv[i].fillna(combine_2_pv[i].mean(), inplace=True)

# # Boxplot Average Score by Prefix
# plt.figure(figsize=(14, 6))
# sns.boxplot(x='subject_code', y='average_score', hue='status', data=combine_df)
# plt.title('Boxplot Average Score by Prefix')
# plt.xlabel('Prefix')
# plt.ylabel('Average Score')
# plt.legend(title='Status')
# plt.show()

cg_df = combine_df.groupby('student_code').agg({
    'attendance_rate': 'mean',
    'number_of_credit': 'sum',
    'total_score': 'sum'
}).reset_index()

cg_df['average_score'] = cg_df['total_score'] / cg_df['number_of_credit']

merged_df = pd.merge(combine_data, cg_df, on='student_code')

merged_df = merged_df[['student_code', 'status', 'attendance_rate',
                       'average_score']]

merged_1 = pd.merge(combine_1_pv, merged_df, on=['student_code', 'status'])
merged_pv = pd.merge(merged_1, combine_2_pv, on=['student_code', 'status'])
merged_pv['status'] = merged_pv['status'].apply(\
    lambda x: 1 if x == 'THO' else 0)
merged_pv.drop('student_code', axis=1, inplace=True)

# # Scatterplot Average Score vs Attendance Percentage (tổng hợp)
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x='average_score', y='attendance_rate',
#                 hue='status', data=merged_df)
# plt.title('Scatter Plot of Average Score vs Attendance Percentage')
# plt.xlabel('Average Score')
# plt.ylabel('Attendance Percentage')
# plt.show()

# # Boxplot Average Score (tổng hợp)
# plt.figure(figsize=(10, 10))
# sns.boxplot(x='status', y='average_score', data=merged_df)
# plt.title('Boxplot of Average Score')
# plt.xlabel('Status')
# plt.ylabel('Average Score')
# plt.show()

# # Histogram Average Score (tổng hợp)
# plt.figure(figsize=(10, 10))
# sns.histplot(x='average_score', hue='status', data=merged_df, kde=True)
# plt.title('Histogram of Average Score')
# plt.xlabel('Average Score')
# plt.ylabel('Frequency')
# plt.show()

# # Cumulative Distribution Average Score
# plt.figure(figsize=(10, 6))
# sns.ecdfplot(data=merged_df, x='average_score', hue='status')
# plt.title('Cumulative Distribution Plot of Average Score: THO vs HDI')
# plt.xlabel('Average Score')
# plt.ylabel('Cumulative Probability')
# plt.show()

# # Boxplot Attendance Rate (tổng hợp)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='status', y='attendance_rate', data=merged_df)
# plt.title('Boxplot of Attendance Percentages: THO vs HDI')
# plt.xlabel('Status Group')
# plt.ylabel('Attendance Percentage')
# plt.show()

# # Histogram Attendance Rate (tổng hợp)
# plt.figure(figsize=(10, 6))
# sns.histplot(data=merged_df, x='attendance_rate', hue='status', kde=True)
# plt.title('Histogram of Attendance Percentages: THO vs HDI')
# plt.xlabel('Attendance Percentage')
# plt.ylabel('Frequency')
# plt.show()

# # Cumulative Distribution Attendance Rate
# plt.figure(figsize=(10, 6))
# sns.ecdfplot(data=merged_df, x='attendance_rate', hue='status')
# plt.title('Cumulative Distribution Plot of Attendance Percentages: THO vs HDI')
# plt.xlabel('Attendance Percentages')
# plt.ylabel('Cumulative Probability')
# plt.show()

merged_df['status'] = merged_df['status'].apply(\
    lambda x: 1 if x == 'THO' else 0)
merged_df.drop('student_code', axis=1, inplace=True)

# minority_class = merged_df[merged_df['status'] == 1]
# majority_class = merged_df[merged_df['status'] == 0]
# # Downsample the majority class
# majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
# balanced_data = pd.concat([majority_downsampled, minority_class])

# minority_class = merged_df[merged_df['status'] == 1]
# majority_class = merged_df[merged_df['status'] == 0]
# # Upsample the minority class
# minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class)/2, random_state=42)
# balanced_data = pd.concat([minority_upsampled, majority_class])

"""
After tuning
"""
X_train_val,X_test,y_train_val,y_test = train_test_split(merged_pv.drop(['status'], axis=1),
                                                        merged_pv['status'], 
                                                        test_size=0.2, random_state=42)

X_train,X_val,y_train,y_val = train_test_split(X_train_val, y_train_val,
                                               test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

param_grid = {'lr__C': [0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(solver='liblinear')

pipeline = Pipeline(steps=[('smote', smote), ('lr', lr)])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='f1', n_jobs=-1)

grid_search.fit(X_train, y_train)
# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Evaluate on the validation set
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Classification report and confusion matrix for validation set
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = best_model.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print('Precision: ', precision_score(y_test, y_test_pred))
print('Recall: ', recall_score(y_test, y_test_pred))
print('F1: ', f1_score(y_test, y_test_pred))

sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d')
plt.title(f'Logistic Regression Accuracy Score: {accuracy_score(y_test, y_test_pred):.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC AUC calculation and plotting
y_test_prob = best_model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

"""
Random Forest
"""
param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

pipeline_rf = Pipeline(steps=[('smote', smote), ('rf', rf)])

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=kfold, scoring='f1', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best hyperparameters for Random Forest
best_params_rf = grid_search_rf.best_params_
print(f"Best Hyperparameters for Random Forest: {best_params_rf}")

# Best model for Random Forest
best_model_rf = grid_search_rf.best_estimator_
y_val_pred_rf = best_model_rf.predict(X_val)

# Classification report and confusion matrix for validation set
print("Validation Set Evaluation")
print(confusion_matrix(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))

# Evaluate on the test set
y_test_pred_rf = best_model_rf.predict(X_test)

# Evaluate the model on the test set
print("Accuracy:", accuracy_score(y_test, y_test_pred_rf))
print('Precision: ', precision_score(y_test, y_test_pred_rf))
print('Recall: ', recall_score(y_test, y_test_pred_rf))
print('F1: ', f1_score(y_test, y_test_pred_rf))

sns.heatmap(confusion_matrix(y_test, y_test_pred_rf), annot=True, fmt='d')
plt.title(f'Random Forest Accuracy Score: {accuracy_score(y_test, y_test_pred_rf):.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

y_test_prob = best_model_rf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

"""
Voting Regression
"""

voting_clf = VotingClassifier(estimators=[
    ('lr', best_model),
    ('rf', best_model_rf)
], voting='soft', n_jobs=-1)

voting_clf.fit(X_train, y_train)

# Evaluate the Voting Classifier on the validation set
y_val_pred_voting = voting_clf.predict(X_val)

print("Validation Set Evaluation with Voting Classifier")
print(confusion_matrix(y_val, y_val_pred_voting))
print(classification_report(y_val, y_val_pred_voting))

# Evaluate the Voting Classifier on the test set
y_test_pred_voting = voting_clf.predict(X_test)

print("Test Set Evaluation with Voting Classifier")
print("Accuracy:", accuracy_score(y_test, y_test_pred_voting))
print('Precision: ', precision_score(y_test, y_test_pred_voting))
print('Recall: ', recall_score(y_test, y_test_pred_voting))
print('F1: ', f1_score(y_test, y_test_pred_voting))

sns.heatmap(confusion_matrix(y_test, y_test_pred_voting), annot=True, fmt='d')
plt.title(f'Voting Classifier Accuracy Score: {accuracy_score(y_test, y_test_pred_voting):.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
Before tuning
"""
X_train, X_test, y_train, y_test = train_test_split(merged_pv.drop(['status'], axis=1),
                                                    merged_pv['status'],
                                                    test_size=0.2, random_state=42)
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title(f'Without Tuning Accuracy Score: {accuracy_score(y_test, y_pred):.4f}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1: ', f1_score(y_test, y_pred))

# pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['attendance_rate', 'average_score'])
# pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
# pivot_df.reset_index(inplace=True)
# pivot_df['status'] = pivot_df['status'].apply(\
#     lambda x: 0 if x == 'THO' else 1)
# # pivot_df.drop('student_code', axis=1, inplace=True)

# join_df = pd.merge(merged_df, pivot_df, on=['student_code', 'status'])
# join_df.drop(['student_code'], axis=1, inplace=True)
# join_df = join_df.fillna(join_df.mean())

# # Correlation Map tổng hợp
# plt.figure(figsize=(10, 10))
# sns.heatmap(merged_df.corr(), annot=True, cmap='crest', linewidth=.1)
# plt.title('Correlation Heatmap')
# plt.show()

# X_train,X_test,y_train,y_test = train_test_split(merged_df.drop(['status'], axis=1),
#                                                  merged_df['status'], 
#                                                  test_size=0.4, random_state=42)
# lr = LogisticRegression()

# model = SVC() 
# model.fit(X_train, y_train) 
  
# # print prediction results 
# predictions = model.predict(X_test) 
# print(classification_report(y_test, predictions))

# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']}  
  
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# # fitting the model for grid search 
# grid.fit(X_train, y_train)

# print(grid.best_params_) 
  
# # print how our model looks after hyper-parameter tuning 
# print(grid.best_estimator_) 

# grid_predictions = grid.predict(X_test) 
  
# # print classification report 
# print(classification_report(y_test, grid_predictions)) 

# kf = KFold(n_splits=2, shuffle=True, random_state=42)  # You can adjust the number of splits as needed

# # Perform K-fold cross-validation and get scores
# cv_scores = cross_val_score(lr, merged_df.drop(['status'], axis=1), merged_df['status'], cv=kf)

# # Print the cross-validation scores
# print("Cross-validation scores:", cv_scores)
# print("Mean CV accuracy:", cv_scores.mean())

# fold = 1
# for train_index, test_index in kf.split(merged_df):
#     X_train, X_test = merged_df.drop(['status'], axis=1).iloc[train_index], merged_df.drop(['status'], axis=1).iloc[test_index]
#     y_train, y_test = merged_df['status'].iloc[train_index], merged_df['status'].iloc[test_index]
    
#     lr.fit(X_train, y_train)
#     y_pred = lr.predict(X_test)
    
#     print(f"\nFold {fold}:")
#     print('Accuracy Score: ', accuracy_score(y_test, y_pred))
#     print('Precision Score: ', precision_score(y_test, y_pred))
#     print('Recall Score: ', recall_score(y_test, y_pred))
#     print('F1 Score: ', f1_score(y_test, y_pred))
    
#     # Plot confusion matrix
#     sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
#     plt.title(f'Fold {fold} - Accuracy Score: {accuracy_score(y_test, y_pred)}')
#     plt.ylabel('Predicted')
#     plt.xlabel('Actual')
#     plt.show()
    
#     fold += 1

# lr.fit(X_train,y_train)
# y_pred = lr.predict(X_test)
# print('Predicted values:\n', y_pred[:10])
# print('Actual values:\n', y_test[:10])
# print(lr.score(X_test,y_test))

# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
# plt.title('Accuracy Score: {}'.format(accuracy_score(y_test, y_pred)))
# plt.ylabel('Predicted')
# plt.xlabel('Actual')
# plt.show()
# print('Logistic Regression')
# print('Accuracy Score: ', accuracy_score(y_test, y_pred))
# print('Precision Score: ', precision_score(y_test, y_pred))
# print('Recall Score: ', recall_score(y_test, y_pred))
# print('F1 Score: ', f1_score(y_test, y_pred))



# # combine_df1 = pd.concat(df_arr1)
# # combine_df2 = pd.concat(df_arr2)
# # combine_df3 = pd.concat(df_arr3)
# # ind_com = [combine_df1, combine_df2, combine_df3]

# # for idx, cdf in enumerate(ind_com):
# #     cdf = cdf.dropna()
# #     cdf["attendance_rate"] = cdf["attendance_rate"].astype(float)
# #     cdf["average_score"] = cdf["average_score"].astype(float)
# #     plt.figure(figsize=(14, 6))
# #     sns.boxplot(x='subject_code', y='average_score', hue='status', data=cdf)
# #     plt.title(f'Boxplot Average Score Prefix Semester {idx}')
# #     plt.xlabel('Subject Code')
# #     plt.ylabel('Average Score')
# #     plt.legend(title='Status')
# #     plt.show()

# unique_subjects = combine_df['subject_code'].unique()

# for subject in unique_subjects:
#     subject_data = combine_df[combine_df['subject_code'] == subject]
#     fig, axs = plt.subplots(1, 2, figsize=(20, 10))
#     sns.scatterplot(x='attendance_rate', y='average_score', hue='status', data=subject_data, ax=axs[0])
#     sns.histplot(data=subject_data, x='average_score', ax=axs[1], kde=True, hue='status')
#     fig.suptitle(f'Subject {subject}')
#     plt.show()

# # Correlation Map Attendance Rate by Prefix only
# pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['attendance_rate'])
# pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
# pivot_df.reset_index(inplace=True)
# pivot_df['status'] = pivot_df['status'].apply(\
#     lambda x: 0 if x == 'THO' else 1)
# pivot_df.drop('student_code', axis=1, inplace=True)
# plt.figure(figsize=(30, 30))
# sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
# plt.title('Correlation Heatmap')
# plt.show()

# # Correlation Map Average Score by Prefix only
# pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['average_score'])
# pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
# pivot_df.reset_index(inplace=True)
# pivot_df['status'] = pivot_df['status'].apply(\
#     lambda x: 0 if x == 'THO' else 1)
# pivot_df.drop('student_code', axis=1, inplace=True)
# plt.figure(figsize=(30, 30))
# sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
# plt.title('Correlation Heatmap')
# plt.show()

# # Correlation Map of both Attendance Rate and Average Score by Prefix
# pivot_df = combine_df.pivot(index=['student_code', 'status'], columns='subject_code', values=['attendance_rate', 'average_score'])
# pivot_df.columns = ['{}_{}'.format(col[0], col[1]) for col in pivot_df.columns]
# pivot_df.reset_index(inplace=True)
# pivot_df['status'] = pivot_df['status'].apply(\
#     lambda x: 0 if x == 'THO' else 1)
# pivot_df.drop('student_code', axis=1, inplace=True)
# plt.figure(figsize=(30, 30))
# sns.heatmap(pivot_df.corr(), annot=True, cmap='crest', linewidth=.1)
# plt.title('Correlation Heatmap')
# plt.show()


