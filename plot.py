import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("attendance_ut2.csv")
df1 = df1[["User_Login", "Percentage"]]

df2 = pd.read_csv("avg_grade_ut2.csv")
df2 = df2[["User_Login", "Average_Grade", "Fail_Count"]]

df3 = pd.read_csv("avg_grade_onlyIT.csv")
df3 = df3[["User_Login", "Avg_COM", "Fail_COM", "Avg_MOB", "Fail_MOB", "Avg_PRO", "Fail_PRO", "Avg_WEB", "Fail_WEB"]]

df = pd.read_csv("status.csv")

merge_df = pd.merge(df1, df, on="User_Login")
plt.figure(figsize=(10,6))
sns.histplot(data=merge_df, x="Percentage", hue="Status", multiple="stack", bins=20)
plt.title('Attendance - Dropout')
plt.xlabel("Attendance rate")
plt.ylabel("Number of dropout")
plt.show()

merge_df = pd.merge(df1, df, on="User_Login", how="inner")
plt.figure(figsize=(10, 6))
sns.boxplot(data=merge_df, x="Semester", y="Percentage", hue="Status")
plt.title("Attendance - Dropout")
plt.xlabel("Semester")
plt.ylabel("Attendance")
plt.legend(title="Status")
plt.show()

merge_df = pd.merge(df2, df, on="User_Login", how="inner")
plt.figure(figsize=(10, 6))
sns.boxplot(data=merge_df, x="Semester", y="Average_Grade", hue="Status")
plt.title("Average_grade - Dropout")
plt.xlabel("Semester")
plt.ylabel("Average_grade")
plt.legend(title="Status")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=merge_df, x="Semester", y="Fail_Count", hue="Status")
plt.title("Fail_count - Dropout")
plt.xlabel("Semester")
plt.ylabel("Fail_count")
plt.legend(title="Status")
plt.show()

merge_df = pd.merge(df2, df, on="User_Login", how='inner')
merged_df = merge_df[merge_df['Semester'] <= 3]
sns.pairplot(merged_df, hue='Status', diag_kind='kde', markers=["o", "s"])

merge_df = pd.merge(df3, df, on="User_Login", how='inner')
merged_df = merge_df[merge_df['Semester'] <= 3]
sns.pairplot(merged_df, hue='Status', diag_kind='kde', markers=["o", "s"])

status_mapping = {status: idx for idx, status in enumerate(df['Status'].unique())}
df['Status_Code'] = df['Status'].map(status_mapping)

subject_related_columns = [col for col in df3.columns if 'Fail_' in col or 'Avg_' in col]

correlation_df = pd.merge(df3, df[['User_Login', 'Status_Code']], on="User_Login")
correlation_df = correlation_df.drop('User_Login', axis=1)

correlation_matrix = correlation_df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap between Subject Performance and Test Status')
plt.show()

