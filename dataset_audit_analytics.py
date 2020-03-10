import pandas as pd

#load the file from Audit Analytics
csv_file1 = "restatements_audit_analytics.csv"
csv_data1 = pd.read_csv(csv_file1, low_memory = False)
csv_df1 = pd.DataFrame(csv_data1)

#this csv has only one column, there are years, rics, and effects in this column, I need to split it into 3 columns
csv_df1['year'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[0])
csv_df1['ric'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[1])
csv_df1['effect'] = csv_df1['year;ric;effect'].map(lambda x:x.split(';')[2])

csv_df1['year']=csv_df1['year'].astype(int)

print(csv_df1)

#load the second file. In this file, there are data from TR Eikon
csv_file2 = "testdateimitlabels.csv"
csv_data2 = pd.read_csv(csv_file2, low_memory = False)
csv_df2 = pd.DataFrame(csv_data2)
print(csv_df2)

#merge two data frames according to ric and year
csv_fusion = pd.merge(csv_df2 , csv_df1 , how='left', on=['ric','year'])

#because both two files have columns "year", "ric" and "effect", we delete extra columns, make the data frame clean
csv_audit = csv_fusion.drop(['year;ric;effect'], axis=1)

# delete duplicate rows, there are duplicat data from Audic Analytics
csv_audit.drop_duplicates(subset=None, keep='first', inplace=True)

print(csv_audit)

#Some firm-years are both negative and positive, we found all these firm-years and saved in d_rows
d_rows=csv_audit[csv_audit.duplicated(subset=['ric','year'],keep=False)]


##########################################################
#Here we want to creat a csv, when a firm-year's effect is negative, then we marked this firm-year as 1.

# For firm-years (positive and negative), we marked also 1. We print all rows with negative effect among
# these firm-years (positive and negative), and saved in d_rows_negative
d_rows_negative= d_rows[d_rows['effect']=='negative']
print(d_rows_negative)

# We deleted all firm-years which are positve and negative.
csv_audit.drop_duplicates(subset=['ric','year'], keep=False, inplace=True)

# We appended only negative rows of these firm-years (positive and negative) to the csv_audit, and now
# firm-years (negative and positive) change into firm-years(negative)
csv_audit_negative=csv_audit.append(d_rows_negative)

#replaced negative with 1, and replaced positive with 0. For the firm-years, which are not covered by Audic Analytics,
#are also marked as 0
csv_audit_negative['effect'].replace('negative','1',inplace=True)

csv_audit_negative['effect'].replace('positive','0',inplace=True)

csv_audit_negative['effect'].fillna(0,inplace=True)

print(csv_audit_negative)

csv_audit_negative.to_csv('dataset_negative_audit_analytics.csv',index=False)


###########################################################
#Here we want to creat a csv, when a firm-year's effect is positive, then we marked this firm-year as 1.

# For firm-years (positive and negative), we marked also 1. We print all rows with positive effect among
# these firm-years (positive and negative), and saved in d_rows_positive
d_rows_positive= d_rows[d_rows['effect']=='positive']
print(d_rows_positive)

# We appended only positive rows of these firm-years (positive and negative) to the csv_audit, and now
# firm-years (negative and positive) change into firm-years(positive)
csv_audit_positive=csv_audit.append(d_rows_positive)

#replaced negative with 0, and replaced positive with 1. For the firm-years, which are not covered by Audic Analytics,
#are also marked as 0

csv_audit_positive['effect'].replace('negative','0',inplace=True)

csv_audit_positive['effect'].replace('positive','1',inplace=True)

csv_audit_positive['effect'].fillna(0,inplace=True)

print(csv_audit_positive)

csv_audit_positive.to_csv('dataset_positive_audit_analytics.csv',index=False)

##########################################################
#Here we want to creat a csv, when a firm-year's effect is positive or positive, then we marked this firm-year as 1.

# For firm-years (positive and negative), we can append positive rows or negative rows
csv_audit_negative_positive = csv_audit.append(d_rows_positive)

#replaced negative with 1, and replaced positive with 1. For the firm-years, which are not covered by Audic Analytics,
#are also marked as 0
csv_audit_negative_positive['effect'].replace('negative','1',inplace=True)

csv_audit_negative_positive['effect'].replace('positive','1',inplace=True)

csv_audit_negative_positive['effect'].fillna(0,inplace=True)

print(csv_audit_negative_positive)

csv_audit_negative_positive.to_csv('dataset_negative_positive_audit_analytics.csv',index=False)