import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

attendance_data = pd.read_csv("attendance_update.csv")
status_data = pd.read_csv("status_update.csv")
status_data_semester_3 = status_data[status_data['Semester'] == 3]

merged_attendance_status = pd.merge(attendance_data, status_data_semester_3,
                                    left_on='User_Code', right_on='Student_Code')
merged_attendance_status["Status_Group"] = merged_attendance_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'Others'
)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Status_Group', y='Percentage', data=merged_attendance_status)
plt.title('Boxplot of Attendance Percentages: THO vs Others')
plt.xlabel('Status Group')
plt.ylabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_attendance_status, x='Percentage', hue='Status_Group', kde=True)
plt.title('Histogram of Attendance Percentages: THO vs Others')
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
plt.title('Density Plot of Attendance Percentages: THO vs Others')
plt.xlabel('Attendance Percentage')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_attendance_status, x='Percentage', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Attendance Percentages: THO vs Others')
plt.xlabel('Attendance Percentages')
plt.ylabel('Cumulative Probability')
plt.show()

avg_grade_data = pd.read_csv("avg_grade_update.csv")

merged_avg_status = pd.merge(avg_grade_data, status_data_semester_3,
                                    left_on='User_Code', right_on='Student_Code')
merged_avg_status["Status_Group"] = merged_avg_status["Status"].apply(
    lambda x: 'THO' if x == 'THO' else 'Others'
)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Status_Group', y='Average_Grade', data=merged_avg_status)
plt.title('Boxplot of Average Grades: THO vs Others')
plt.xlabel('Status Group')
plt.ylabel('Average Grades')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group', kde=True)
plt.title('Histogram of Average Grades: THO vs Others')
plt.xlabel('Average Grades')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group', fill=True)
plt.title('Density Plot of Average Grades: THO vs Others')
plt.xlabel('Average Grades')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_avg_status, x='Average_Grade', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Average Grades: THO vs Others')
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

merged_grade_prefix_status_data['Status_Group'] = merged_grade_prefix_status_data['Status'].apply(lambda x: 'THO' if x == 'THO' else 'Others')

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
plt.title('Histogram of Average Grades by Status Group')
plt.xlabel('Average Grade')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Prefix_Subject', y='Average_Grade', hue='Status_Group', data=merged_grade_prefix_status_data)
plt.title('Scatter Plot of Semester vs Average Grade by Status Group')
plt.xlabel('Prefix_Subject')
plt.ylabel('Average Grade')
plt.show()

review_data = pd.read_csv("review.csv")

type_counts = review_data.groupby(['User_Code', 'Type']).size().unstack(fill_value=0)

type_counts.head()

type_percentages = type_counts.div(type_counts.sum(axis=1), axis=0)

type_percentages_reset = type_percentages.reset_index()

merged_review_status = pd.merge(type_percentages_reset, status_data_semester_3, left_on='User_Code', right_on='Student_Code')

merged_review_status['Status_Group'] = merged_review_status['Status'].apply(lambda x: 'THO' if x == 'THO' else 'Others')

average_type_by_status = merged_review_status.groupby('Status_Group')[[0, 1, 2, 3]].mean()

average_type_by_status_melted = average_type_by_status.reset_index().melt(id_vars='Status_Group', var_name="Type", value_name="Average Percentage")

plt.figure(figsize=(12, 6))
sns.barplot(average_type_by_status_melted, x="Type", y="Average Percentage", hue="Status_Group")
plt.xlabel('Type')
plt.ylabel('Average Percentage')
plt.title('Average Percentage of Each Student Type by Status in Semester 3')
plt.legend()
plt.show()

fail_data = pd.read_csv('fail_count.csv')

merged_fail_status = pd.merge(fail_data, status_data_semester_3, left_on='User_Code', right_on='Student_Code')

merged_fail_status["Status_Group"] = merged_fail_status["Status"].apply(lambda x: 'THO' if x == 'THO' else 'Others')

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
plt.title('Density Plot of Total Fail: THO vs Others')
plt.xlabel('Total Fail')
plt.show()

plt.figure(figsize=(10, 6))
sns.ecdfplot(data=merged_fail_status, x='Total_Fail', hue='Status_Group')
plt.title('Cumulative Distribution Plot of Total Fail: THO vs Others')
plt.xlabel('Total Fail')
plt.ylabel('Cumulative Probability')
plt.show()

fail_prefix_data = pd.read_csv('fail_count_by_prefix.csv')

merged_fail_prefix_status = pd.merge(fail_prefix_data, status_data_semester_3, left_on='User_Code', right_on='Student_Code')

merged_fail_prefix_status['Status_Group'] = merged_fail_prefix_status['Status'].apply(lambda x: 'THO' if x == 'THO' else 'Others')

plt.figure(figsize=(10, 6))
sns.histplot(data=merged_fail_prefix_status, x='Total_Fail', hue='Status_Group', kde=True)
plt.title('Histogram of Fail Counts by Status Group')
plt.xlabel('Fail Counts')
plt.ylabel('Frequency')
plt.show()
