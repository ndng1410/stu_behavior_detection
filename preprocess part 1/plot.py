import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

attendance_data = pd.read_csv("attendance_update.csv")
status_data = pd.read_csv("status_update.csv")
status_data_semester_3 = status_data[status_data['Semester'] == 3]

merged_attendance_status = pd.merge(attendance_data, status_data_semester_3,
                                    left_on='User_Code', right_on='Student_Code')
merged_attendance_status["Status_Group"] = merged_attendance_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI'
)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Status_Group', y='Percentage', data=merged_attendance_status)
plt.title('Boxplot of Attendance Percentages: THO vs HDI')
plt.xlabel('Status Group')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_attendance_status, x='Percentage', hue='Status_Group', kde=True)
plt.title('Histogram of Attendance Percentages: THO vs HDI')
plt.xlabel('Attendance Percentage')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Activity', y='Percentage', hue='Status_Group', data=merged_attendance_status)
plt.title('Scatter Plot of Total Activity vs Attendance Percentage')
plt.xlabel('Total Activity')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=merged_attendance_status, x='Percentage', hue='Status_Group', fill=True)
plt.title('Density Plot of Attendance Percentages: THO vs HDI')
plt.xlabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_attendance_status, x='Percentage', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Attendance Percentages: THO vs HDI')
plt.xlabel('Attendance Percentages')
plt.ylabel('Cumulative Probability')
plt.show()

avg_grade_data = pd.read_csv("avg_grade_update.csv")

merged_avg_status = pd.merge(avg_grade_data, status_data_semester_3,
                                    left_on='User_Code', right_on='Student_Code')
merged_avg_status["Status_Group"] = merged_avg_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

plt.figure(figsize=(10, 6))
sns.boxplot(x='Status_Group', y='Average_Grade', data=merged_avg_status)
plt.title('Boxplot of Average Grades: THO vs HDI')
plt.xlabel('Status Group')
plt.ylabel('Average Grades')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group', kde=True)
plt.title('Histogram of Average Grades: THO vs HDI')
plt.xlabel('Average Grades')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y="Total_Credit", hue='Status_Group', data=merged_avg_status)
plt.title('Scatter Plot of Total Credit vs Average Grade')
plt.xlabel('Average Grade')
plt.ylabel('Total Credit')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group', fill=True)
plt.title('Density Plot of Average Grades: THO vs HDI')
plt.xlabel('Average Grades')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Average Grades: THO vs HDI')
plt.xlabel('Average Grades')
plt.ylabel('Cumulative Probability')
plt.show()

combined_data = pd.merge(merged_attendance_status[['User_Code', 'Percentage', 'Status_Group']], 
                         merged_avg_status[['User_Code', 'Average_Grade', 'Status_Group']], 
                         on='User_Code')
combined_data = combined_data[combined_data['Status_Group_x'] == combined_data['Status_Group_y']]
combined_data.drop(columns=['Status_Group_y'], inplace=True)
combined_data.rename(columns={'Status_Group_x': 'Status_Group'}, inplace=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Percentage', y='Average_Grade', hue='Status_Group', data=combined_data)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group')
plt.xlabel('Attendance Percentage')
plt.ylabel('Average Grade')
plt.show()

avg_grade_prefix_data = pd.read_csv("avg_grade_by_prefix_update.csv")

merged_grade_prefix_status_data = pd.merge(avg_grade_prefix_data, status_data_semester_3, left_on='User_Code', right_on='Student_Code')
merged_grade_prefix_status_data['Status_Group'] = merged_grade_prefix_status_data['Status'].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

plt.figure(figsize=(14, 6))
sns.boxplot(x='Prefix_Subject', y='Average_Grade', hue='Status_Group', data=merged_grade_prefix_status_data)
plt.title('Boxplot of Average Grades by Subject Prefix and Status Group')
plt.xlabel('Subject Prefix')
plt.ylabel('Average Grade')
plt.xticks(rotation=45)
plt.legend(title='Status Group')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_grade_prefix_status_data, x='Average_Grade', hue='Status_Group', kde=True)
plt.title('Histogram of Average Grades by Prefix by Status Group')
plt.xlabel('Average Grade')
plt.ylabel('Frequency')
plt.show()

merged_grade_prefix_status_data_MOB = merged_grade_prefix_status_data[merged_grade_prefix_status_data['Prefix_Subject'] == 'MOB']
merged_grade_prefix_status_data_WEB = merged_grade_prefix_status_data[merged_grade_prefix_status_data['Prefix_Subject'] == 'WEB']
merged_grade_prefix_status_data_NET = merged_grade_prefix_status_data[merged_grade_prefix_status_data['Prefix_Subject'] == 'NET']
merged_grade_prefix_status_data_COM = merged_grade_prefix_status_data[merged_grade_prefix_status_data['Prefix_Subject'] == 'COM']
merged_grade_prefix_status_data_ENT = merged_grade_prefix_status_data[merged_grade_prefix_status_data['Prefix_Subject'] == 'ENT']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
sns.histplot(data=merged_grade_prefix_status_data_MOB, x='Average_Grade', ax=axes[0, 0], kde=True, hue='Status_Group')
axes[0, 0].set_title('Average MOB')
sns.histplot(data=merged_grade_prefix_status_data_WEB, x='Average_Grade', ax=axes[0, 1], kde=True, hue='Status_Group')
axes[0, 1].set_title('Average WEB')
sns.histplot(data=merged_grade_prefix_status_data_NET, x='Average_Grade', ax=axes[1, 0], kde=True, hue='Status_Group')
axes[1, 0].set_title('Average NET')
sns.histplot(data=merged_grade_prefix_status_data_COM, x='Average_Grade', ax=axes[1, 1], kde=True, hue='Status_Group')
axes[1, 1].set_title('Average COM')
# sns.histplot(data=merged_grade_prefix_status_data_ENT, x='Average_Grade', ax=axes[1, 1], kde=True, hue='Status_Group')
# axes[1, 1].set_title('Average ENT')
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# sns.pairplot(hue='Status_Group', data=merged_grade_prefix_status_data, vars=['Prefix_Subject', 'Average_Grade'])
# plt.title('scatterplot matrix')
# plt.xlabel('Prefix_Subject')
# plt.ylabel('Average Grade')
# plt.show()

review_data = pd.read_csv("review.csv")

type_counts = review_data.groupby(['User_Code', 'Type']).size().unstack(fill_value=0)
type_percentages = type_counts.div(type_counts.sum(axis=1), axis=0)
type_percentages_reset = type_percentages.reset_index()
merged_review_status = pd.merge(type_percentages_reset, status_data_semester_3, 
                                left_on='User_Code', right_on='Student_Code')
merged_review_status['Status_Group'] = merged_review_status['Status'].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')
average_type_by_status = merged_review_status.groupby('Status_Group')[[0, 1, 2, 3]].mean()
average_type_by_status_melted = average_type_by_status.reset_index().melt(id_vars='Status_Group', 
                                                                          var_name="Type", value_name="Average_Percentage")

plt.figure(figsize=(12, 6))
sns.barplot(average_type_by_status_melted, x="Type", y="Average_Percentage", hue="Status_Group")
plt.xlabel('Type')
plt.ylabel('Average Percentage')
plt.title('Average Percentage of Each Student Type by Status in Semester 3')
plt.legend()
plt.show()

merged_review_status_x = pd.merge(review_data, status_data_semester_3, 
                                  left_on='User_Code', right_on='Student_Code')
merged_review_status_x['Status_Group'] = merged_review_status_x['Status'].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')
merged_review_status_x = merged_review_status_x[merged_review_status_x['Type'] != 3]

plt.figure(figsize=(10, 6))
sns.countplot(merged_review_status_x, x='Type', hue='Status_Group')
plt.title('Counts of Student Type 0 - 1 - 2')
plt.show()

fail_data = pd.read_csv('fail_count.csv')

merged_fail_status = pd.merge(fail_data, status_data_semester_3, 
                              left_on='User_Code', right_on='Student_Code')
merged_fail_status["Status_Group"] = merged_fail_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

plt.figure(figsize=(10, 6))
sns.boxplot(merged_fail_status, x='Status_Group', y='Total_Fail')
plt.title('Boxplot of Total Fail by Status Group')
plt.xlabel('Total Fail')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(merged_fail_status, x='Total_Fail', hue='Status_Group', kde=True)
plt.title('Histogram of Total Fail by Status Group')
plt.xlabel('Total Fail')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=merged_fail_status, x='Total_Fail', hue='Status_Group', fill=True)
plt.title('Density Plot of Total Fail: THO vs HDI')
plt.xlabel('Total Fail')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_fail_status, x='Total_Fail', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Total Fail: THO vs HDI')
plt.xlabel('Total Fail')
plt.ylabel('Cumulative Probability')
plt.show()

fail_prefix_data = pd.read_csv('fail_count_by_prefix.csv')

merged_fail_prefix_status = pd.merge(fail_prefix_data, status_data_semester_3, 
                                     left_on='User_Code', right_on='Student_Code')
merged_fail_prefix_status['Status_Group'] = merged_fail_prefix_status['Status'].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_fail_prefix_status, x='Total_Fail', hue='Status_Group', kde=True)
plt.title('Histogram of Fail Counts by Status Group')
plt.xlabel('Fail Counts')
plt.ylabel('Frequency')
plt.show()

attendance_prefix = pd.read_csv("attendance_prefix.csv")

merged_attpre_status = pd.merge(attendance_prefix, status_data_semester_3,
                                    left_on='User_Code', right_on='Student_Code')
merged_attpre_status["Status_Group"] = merged_attpre_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'HDI')
merged_attpre_status[['User_Code', 'Prefix', 'Percentage', 'Status_Group']]
merged_grade_prefix_status_data[['User_Code', 'Average_Grade', 'Prefix_Subject', 'Status_Group']]

combined_data = pd.merge(merged_attpre_status[['User_Code', 'Prefix', 'Percentage', 'Status_Group']], 
                         merged_grade_prefix_status_data[['User_Code', 'Average_Grade', 'Prefix_Subject', 'Status_Group']], 
                         how='left', 
                         left_on=['User_Code', 'Prefix'],
                         right_on=['User_Code', 'Prefix_Subject'])
combined_data = combined_data[combined_data['Status_Group_x'] == combined_data['Status_Group_y']]
combined_data.drop(columns=['Status_Group_y'], inplace=True)
combined_data.rename(columns={'Status_Group_x': 'Status_Group'}, inplace=True)
combined_data = combined_data[combined_data['Prefix'] == combined_data['Prefix_Subject']]
combined_data.drop(columns=['Prefix_Subject'], inplace=True)

combined_data_MOB = combined_data[combined_data['Prefix'] == 'MOB']
combined_data_WEB = combined_data[combined_data['Prefix'] == 'WEB']
combined_data_NET = combined_data[combined_data['Prefix'] == 'NET']
combined_data_COM = combined_data[combined_data['Prefix'] == 'COM']
combined_data_ENT = combined_data[combined_data['Prefix'] == 'ENT']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y='Percentage', hue='Status_Group', data=combined_data_MOB)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group MOB')
plt.xlabel('Average Grade')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y='Percentage', hue='Status_Group', data=combined_data_WEB)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group WEB')
plt.xlabel('Average Grade')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y='Percentage', hue='Status_Group', data=combined_data_NET)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group NET')
plt.xlabel('Average Grade')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y='Percentage', hue='Status_Group', data=combined_data_COM)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group COM')
plt.xlabel('Average Grade')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Grade', y='Percentage', hue='Status_Group', data=combined_data_ENT)
plt.title('Scatter Plot of Attendance Percentage vs Average Grade by Status Group ENT')
plt.xlabel('Average Grade')
plt.ylabel('Attendance Percentage')
plt.show()

combine_data = pd.merge(attendance_data, status_data_semester_3,
                        left_on='User_Code', right_on='Student_Code')
combine_data = pd.merge(combine_data, avg_grade_data,
                        left_on='User_Code', right_on='User_Code')
combine_data = pd.merge(combine_data, fail_data,
                        left_on='User_Code', right_on='User_Code')
combine_data = combine_data[["Percentage", 
                             "Status", "Average_Grade", 'Total_Fail']]
combine_data['Status'] = combine_data['Status'].apply(
    lambda x: 0 if x == 'THO' else 1)

plt.figure(figsize=(15, 10))
sns.heatmap(combine_data.corr(), annot=True, cmap='crest', linewidth=.5)
plt.title('Correlation Heatmap')
plt.show()


# df = pd.read_csv('test.csv')
# df['Status'] = df['Value'].apply(lambda x: 'P' if x == 1 else 'F')
# df['Subject_Status'] = df["Subject_Code"] + ' - ' + df['Status'] + ' - ' + df['Semester'].astype(str)

# df_concat = df.groupby(['User_Login', 'User_Code', 'Skill_Code']).agg({
#         'Subject_Status': lambda x: ', '.join(x),
#         'Total_Grade': lambda x: sum(x),
#         'Total_Credit': lambda x: sum(x),
#     }).reset_index()
# df_concat["Average_Grade"] = (df_concat['Total_Grade'] / df_concat["Total_Credit"]).round(2)
# # df_concat.head()
# # print(df_concat[df_concat["User_Login"] == 'truongpnph12690'])
# df_concat.to_csv('average_by_skillcode.csv', sep=',', encoding='utf-8')