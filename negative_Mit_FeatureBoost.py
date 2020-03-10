import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import IsolationForest

data = pd.read_csv('dataset_negative_audit_analytics.csv',low_memory=False)

col_names = data.columns
col_list = col_names.tolist()
#Get all keys except years, rics and labels(all, relevant, relevant5%)
keys_X = []
for x in range(5,839):
    keys_X.append(col_list[x])
# get all datas except years,rics and labels
X = data[keys_X]
#Delete all boolean type or object type(datum) features
X = X.select_dtypes(exclude=['bool','object'])
#Replace all the NAN with mean
imp = SimpleImputer(strategy="mean")
X = imp.fit_transform(X)
#Transform the data as data frame
X = pd.DataFrame(X)

X.info()
#get all the labels
y = data[['effect']]

#Using XGBoost for featureboost, each feature get a score.
#Using SelectFromModel, we can extract features, whose score is higher than average score.

model = XGBClassifier()
model.fit(X, y.values.ravel())
selection = SelectFromModel(model,prefit=True,threshold="mean")
select_X = selection.transform(X)
select_X = pd.DataFrame(select_X)

#split training data and test data, The ratio is 4: 1
X_train, X_test, y_train, y_test = train_test_split(select_X, y, test_size=0.2, random_state=0)

#############################################
#ADABOOST
#############################################

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'algorithm':["SAMME","SAMME.R"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(adaBoost(),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy of Adaboost is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

##############################################
#Decison Tree
##############################################
#cross validation and grid search for hyperparameter estimation
param_dist = {
        'splitter' :['best','random']
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(DT(max_depth=1),param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for relevant5% is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

##############################################
#KNN
##############################################

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'weights':["uniform","distance"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(kNN(), param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)
#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for KNN is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))

#################################################
#random forest
#################################################

#cross validation and grid search for hyperparameter estimation
param_dist = {
        'criterion':["gini","entropy"]
        }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
clf = GridSearchCV(RandomForestClassifier(max_features=1), param_grid=param_dist, cv=cv)
clf = clf.fit(X_train, y_train.values.ravel())

print("Best estimator found by grid search:")
print(clf.best_estimator_)

#apply the classifier on the test data and show the accuracy of the model
print('the acuracy for Random Forest is:')
print(clf.score(X_test, y_test.values.ravel()))

prediction = clf.predict(X_test)
#use the metrics.classification to report.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
print("Classification report:\n %s\n"% metrics.classification_report(y_test, prediction))






####################################################
#Isolation Forest
####################################################

y = pd.DataFrame(y)

sel_X = pd.concat([y,select_X],axis=1)
sel_X.info()
print("selX")
print(sel_X)

#get all datas with label "effect" is 1
X_negative1 = sel_X.loc[sel_X["effect"] == 1]
#get all datas with label "effect" is 0
X_negative0 = sel_X.loc[sel_X["effect"] == 0]

#get all data with label "effect" is 1 except label
X_negative_1 = X_negative1.iloc[:, 1:]
print("X_negative_1##########################################")
print(X_negative_1)
#get all data with label "effect" is 0 except label
X_negative_0 = X_negative0.iloc[:, 1:]
print("X_negative_0###########################################")
print(X_negative_0)

X_negative_1.info()
X_negative_0.info()

#set training data,isolation Forest is a semi-supervised algorithm, all training data is normal, and we set 4/5 data set
#as training data
X0_train = X_negative_0.loc[0:109196]
print("X0_train############################################")
print(X0_train)
#set test data,the rest of normal data as test set
X0_test = X_negative_0.loc[109196:]
print("X0_test############################################")
print(X0_test)
#create a classifier
clf = IsolationForest(contamination=0.22)
clf.fit(X0_train)
##use this classifier to predict outliers and test data
y_pred_test = clf.predict(X0_test)
y_pred_outliers = clf.predict(X_negative_1)

# print for a confusion matrix and report.
print("amount of target is  0 and prediction is also 0:")
a00 = list(y_pred_test).count(1)
print(a00)

print("amount of target is  0 and prediction is 1:")
a01 = list(y_pred_test).count(-1)
print(a01)

print("amount of target is  1 and prediction is also 1:")
a11 = list(y_pred_outliers).count(-1)
print(a11)

print("amount of target is  1 and prediction is 0:")
a10 = list(y_pred_outliers).count(1)
print(a10)

print("amount of normal test data")
anormal =y_pred_test.shape[0]
print(anormal)

print("amount of test outliers ")
aoutlier = y_pred_outliers.shape[0]
print(aoutlier)



print("accuracy is")
print((a00+a11)/(anormal+aoutlier))

print("precision of 1 is")
precision_1 = a11/(a11+a01)
print(precision_1)

print("recall of 1 is")
recall_1 = a11/(a11+a10)
print(recall_1)

print("f1score of 1 is")
print((2*recall_1*precision_1)/(recall_1 + precision_1))

print("precision of 0 is")
precision_0 = a00/(a10+a00)
print(precision_0)

print("recall of 0 is")
recall_0 = a00/(a01+a00)
print(recall_0)

print("f1score of 0 is")
print((2*recall_0*precision_0)/(recall_0 + precision_0))




