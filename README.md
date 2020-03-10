# BA_WH


In this project, we used 6 different machine learning algorithms to predict finacial restatements.
The data sets are from TR EIKON and Audit Analytics. So the project consists of 2 parts.

The first part is for data set from TR EIKON:

1. Run restatements.py to get restatements for each firm-year.

2. Run testdatei.py to get data for each firm-year.

3. Run testdateimitlabels.py to combine the restatements and data.

4. Run 6 different algorithms with or without FeatureBoost. These 6 algorithms are in AdaBoost_Mit_FeatureBoost.py, AdaBoost_Ohne_FeatureBoost.py,
DecisionTree_Mit_FeatureBoost.py, DecisionTree_Ohne_FeatureBoost.py, IsolationForest_Mit_FeatureBoost.py, IsolationForest_Ohne_featureBoost.py,
Knn_Mit_FeatureBoost.py, Knn_Ohne_FeatureBoost.py, Randomforest_Mit_FeatureBoost.py, Randomforest_Ohne_FeatureBoost.py,
Svm_Mit_FeatureBoost.py, Svm_Ohne_FeatureBoost.py.

The second part is for data set from Audic Analytics:

1. Run dataset_audit_analytics.py to combine the restatements from Audit Analytics and the data from TR EIKON. After running this program，we
got three csv files. One for label "positive", one for label "negative" and one for label "negative or positive".

2. Run negative_Mit_FeatureBoost.py, negative_Ohne_FeatureBoost.py, svm_negative_Mit_FeatureBoost.py and svm_negative_Ohne_FeatureBoost.py to 
analyse the performance of 6 algorithms regarding "negative". SVM runs separately from other algorithms because svm runs too slowly and needs to be run by HPC.

3. Run positive_Mit_FeatureBoost.py, positive_Ohne_FeatureBoost.py, svm_positive_Mit_FeatureBoost.py and svm_positive_Ohne_FeatureBoost.py to 
analyse the performance of 6 algorithms regarding "positive". SVM runs separately from other algorithms because svm runs too slowly and needs to be run by HPC.

4. Run positive_or_negative_Mit_FeatureBoost.py, positive_or_negative_Ohne_FeatureBoost.py, svm_po_ne_Mit_FeatureBoost.py and svm_po_ne_Ohne_FeatureBoost.py to 
analyse the performance of 6 algorithms regarding "negative or positive". SVM runs separately from other algorithms because svm runs too slowly and needs to be run by HPC.

