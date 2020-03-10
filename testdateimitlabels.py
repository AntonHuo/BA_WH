import pandas as pd

#load the first file, in this file are firm-years restatements
csv_file1 = "restatements.csv"
csv_data1 = pd.read_csv(csv_file1, low_memory = False)
csv_df1 = pd.DataFrame(csv_data1)
#load the second file, in this file are date of each firm-year.
csv_file2 = "testdatei.csv"
csv_data2 = pd.read_csv(csv_file2, low_memory = False)
csv_df2 = pd.DataFrame(csv_data2)
#Merge two dataframes
csv_fusion = pd.concat([csv_df1,csv_df2], axis=1)
#save the new dataframe as a csv file
csv_fusion.to_csv('testdateimitlabels.csv',index=False)