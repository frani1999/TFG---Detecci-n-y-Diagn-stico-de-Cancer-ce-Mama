# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#-----------------------Without bounding features----------------------------------
#df_train = pd.read_csv("train_mass_crops_features.csv")
#df_test = pd.read_csv("test_mass_crops_features.csv")
#----------------------------Bounding features-------------------------------------
df_train = pd.read_csv("train_mass_crop_features_bounded.csv")
df_test = pd.read_csv("test_mass_crop_features_bounded.csv")

#Analyzing the data for null values and dropping the rows having an empty value
#print(df.isnull().sum())
#print(df_train)
#print(df_test)
#print(df.describe())
#print(df.corr())

#We will predict wether the tumor is Malignant or Benign on the basis of radius,texture,smoothness,compactness, and concavity:
x_train = df_train[['area', 'bounding box area', 'convex area', 'exentricity', 'equivalent diameter',
        'extension', 'feret diameter','major axis length','minor axis length', 'orientation',
        'perimeter', 'solidity', 'compactness']]
#print(x_train)
y_train = df_train[['severity']]
y_train = y_train.replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
print(x_train)
print(y_train)

x_test = df_test[['area', 'bounding box area', 'convex area', 'exentricity', 'equivalent diameter',
        'extension', 'feret diameter','major axis length','minor axis length', 'orientation',
        'perimeter', 'solidity', 'compactness']]
y_test = df_test[['severity']]
y_test = y_test.replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
print(x_test)
print(y_test)
#---------------------------------------------------VIEW CORRELATION----------------------------------------------------
corr_x_train = x_train.corr()
corr_x_test = x_test.corr()

print(sns.heatmap(corr_x_train))
ax = sns.heatmap(corr_x_test)
plt.plot(ax)
plt.show()
#---------------------------------------------APPLY MACHINE LEARNING MODEL----------------------------------------------
#KNN:
print("\n\nKNN\n\n")
#Using the model of LogisticRegression:
logmodel = LogisticRegression()

#Training the model and making predictions:
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)

#Checking for the accuracy score using jaccard_similarity_score:
accuracy_score = jaccard_score(y_test,predictions, average='binary', pos_label="BENIGN")
print(accuracy_score*100)
#Giving a look to the confusion matrix:
matrix=confusion_matrix(y_test,predictions)
print(matrix)


#Now, we will be using SVM as our second model and then we will compare the accuracy with the KNN model:
print("\n\nSVC\n\n")
clf = svm.SVC(gamma="scale")
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score = jaccard_score(y_test,predictions, average='binary', pos_label="BENIGN")
print(accuracy_score*100)
#Giving a look to the confusion matrix:
matrix=confusion_matrix(y_test,predictions)
print(matrix)


print("\n\nRandom Forest\n\n")
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy_score = jaccard_score(y_test,predictions, average='binary', pos_label="BENIGN")
print(accuracy_score*100)
#Giving a look to the confusion matrix:
matrix=confusion_matrix(y_test,predictions)
print(matrix)

