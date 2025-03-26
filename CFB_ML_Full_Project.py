#all code in this program is commented out, but feel free to edit out those comments to try things yourself.
#hopefully the comments I provided throughout the code are self-explanatory

import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

csvFile = pd.read_csv("College_Football_Stats.csv")

#first start with FPI prediction
#then use that prediction to predict win rate next year

#all of the above is FPI prediction

#create x and y for FPI prediction
x = csvFile[['2023 FPI', '2024 Recruiting Class', '2024 Returning Production']]
x2 = csvFile[['2023 FPI', '2024 Recruiting Class', '2024 Returning Production']]
y = csvFile[['2024 FPI']]

#dropped x columns (post dimensionality reduction)
#"""
x = x.drop(x.columns[2], axis=1)
#"""

#cross validate linear regression
"""
lr = LinearRegression()
cv_scores = cross_val_score(lr, x, y, cv=4, scoring="neg_mean_squared_error")
for j in range(len(cv_scores)):
    cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
print(cv_scores)
print(round(cv_score,2))
#42.79
"""

#cross validate polynomial regression
"""
for k in range(2, 11, 1):
    poly = PolynomialFeatures(degree=k, include_bias=False)
    poly_x = poly.fit_transform(x)
    pr = LinearRegression()
    cv_scores = cross_val_score(pr, poly_x, y, cv=4, scoring="neg_mean_squared_error")
    for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
    cv_score = np.average(cv_scores)
    cv_score = round(cv_score, 2)
    print(f"Degree={k}: {cv_scores}")
    print(f"{cv_score}")
    #best score is for degree=2, where mse is 41.27
"""

#cross validate knn regression
"""
#change depending on tested p value
print("p=2")
print()
for k in range(1, 24, 2):
    #metric is minkowski for all, just change p from 1 to 3 depending on specific power value
    knr = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
    cv_scores = cross_val_score(knr, x, y, cv=4, scoring="neg_mean_squared_error")
    for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
    cv_score = np.average(cv_scores)
    cv_score = round(cv_score, 2)
    print(f"k={k}: {cv_score}")
"""


#dimensionality reduction (be sure no columns are already dropped from post dimensionality reduction changes)
"""
knr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
cv_scores = cross_val_score(knr, x, y, cv=4, scoring="neg_mean_squared_error")
for j in range(len(cv_scores)):
    cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Starting: {cv_score} stdev: {cv_stdev}")
for k in range(x.shape[1]):
    x2 = x.drop(x.columns[k],axis=1)
    knr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
    cv_scores = cross_val_score(knr, x2, y, cv=4, scoring='neg_mean_squared_error')
    for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
    cv_score = np.average(cv_scores)
    cv_score = round(cv_score, 2)
    cv_stdev = np.std(cv_scores)
    cv_stdev = round(cv_stdev, 2)
    print(f"Dropping column {k+1}: {cv_score} stdev: {cv_stdev}")
    #no reduction needed
"""

#
#Now for part 2, success prediction based on x

#we need to change y since our target value is different
#y1 is to make a bowl game, y2 for making the playoffs
#x2 is the original csv file, because it may not require dimensionality reduction
y1 = csvFile['6+ Wins 2024']
y2 = csvFile['Made Playoffs 2024']

#We can use normal cross_val_score with accuracy with these since they are classifications

#Cross Validate Gaussian Naive Bayes
"""
gnb = GaussianNB()
cv_scores = cross_val_score(gnb, x2, y1, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Gaussian Naive Bayes 6+ Wins: {cv_score}% stdev: {cv_stdev}")
cv_scores = cross_val_score(gnb, x2, y2, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Make Playoffs: {cv_score}% stdev: {cv_stdev}") 
print()
#63.99% accuracy
"""

#Cross Validate Logistic Regression
"""
log = LogisticRegression()
cv_scores = cross_val_score(log, x2, y1, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Logistic Regression: {cv_score}% stdev: {cv_stdev}")
cv_scores = cross_val_score(log, x2, y2, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Make Playoffs: {cv_score}% stdev: {cv_stdev}")
print()
"""


#Cross Validate SVM Classifier (with probability)
"""
#linear
svc = SVC(probability=True, kernel = 'linear')
cv_scores = cross_val_score(svc, x2, y1, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Linear SVC: {cv_score}% stdev: {cv_stdev}")
cv_scores = cross_val_score(svc, x2, y2, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Make Playoffs: {cv_score}% stdev: {cv_stdev}")
print()

#kbf
svc = SVC(probability=True, kernel = 'rbf')
cv_scores = cross_val_score(svc, x2, y1, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Linear SVC: {cv_score}% stdev: {cv_stdev}")
cv_scores = cross_val_score(svc, x2, y2, cv=4)
for j in range(len(cv_scores)):
        cv_scores[j] = abs(cv_scores[j])
cv_score = np.average(cv_scores)
cv_score = cv_score*100
cv_score = round(cv_score, 2)
cv_stdev = np.std(cv_scores)
cv_stdev = round(cv_stdev, 2)
print(f"Make Playoffs: {cv_score}% stdev: {cv_stdev}")
print()
"""


#Part 1 classifier: KNN Regressor (p=2, N_neighbors=5)
#Part 2 classifier: SVM (kernel=rbf) changed to Gaussian NB and then to Logistic Regression

#first a confusion matrix for part 2
"""
#gnb confusion matrix 6+ wins
x_train, x_test, y_train, y_test = train_test_split(x2, y1, stratify=y1, test_size=0.25)
log = LogisticRegression()
log.fit(x2,y1)
y_predict = log.predict(x_test)
#parameters for confusion matrix are (actual, predicted)
cMatrix = metrics.confusion_matrix(y_test, y_predict)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cMatrix, display_labels = ["Less than 6 Wins", "6+ Wins"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
cm_display.plot()
plt.show()


#gnb confusion matrix playoffs
x_train, x_test, y_train, y_test = train_test_split(x2, y2, stratify=y2, test_size=0.25)
log = LogisticRegression()
log.fit(x2,y2)
y_predict = log.predict(x_test)
#parameters for confusion matrix are (actual, predicted)
cMatrix = metrics.confusion_matrix(y_test, y_predict)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cMatrix, display_labels = ["Missed Playoffs", "Made Playoffs"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
cm_display.plot()
plt.show()
"""

#now predicting every FPI and comparing to actual results afterwards, using 2024 FPI
"""
print()
print("Predicted 2024 FPI vs Actual 2024 FPI")
for k in range(len(x)):
    #since we are testing the training set itself, we do not need test_train_split
    knr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
    knr.fit(x,y)
    y_predict = knr.predict(x)
    y_predict = np.round(y_predict, decimals=1)
    print(f"{csvFile['Teams'].iloc[k]}: Predicted: {y_predict[k,0]} Actual: {y['2024 FPI'].iloc[k]}")
    
"""


#Now time to predict for 2025
#columns for training and testing part 1 of 2025 prediction. Models are trained on the 2024 results to predict 2025
x_2025_train = csvFile[['2023 FPI', '2024 Recruiting Class']]
x_2025_test = csvFile[['2024 FPI', '2025 Recruiting Class']]
#renaming features to prevent confusion in the model
x_2025_train.columns = ['FPI', 'Recruiting Class']
x_2025_test.columns = ['FPI', 'Recruiting Class']

y_2025_train = csvFile[['2024 FPI']]

#columns for training and testing part 2 of 2025 prediction. Models are trained on the 2024 results to predict 2025
x2_2025_train = csvFile[['2023 FPI', '2024 Recruiting Class', '2024 Returning Production']]
x2_2025_test = csvFile[['2024 FPI', '2025 Recruiting Class', '2025 Returning Production']]

#renaming features to prevent confusion in the model
x2_2025_train.columns = ['FPI', 'Recruiting Class', 'Returning Production']
x2_2025_test.columns = ['FPI', 'Recruiting Class', 'Returning Production']
#y1 is bowl game chances, y2 is playoff chances
y1_2025_train = csvFile[['6+ Wins 2024']]
y2_2025_train = csvFile[['Made Playoffs 2024']]


#part 1 prediction
"""
#we've already split the data, so no need for train_test_split
knr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
knr.fit(x_2025_train,y_2025_train)
y_predict = knr.predict(x_2025_test)
y_predict = np.round(y_predict, decimals=1)
fpi_predict_list = [[]]
print()
print("Predicted 2025 FPI (Sorted)")
for k in range(len(x)):
    fpi_predict_list.append([csvFile['Teams'].iloc[k], y_predict[k,0]])
    
#removing the first row which is null
fpi_predict_list.remove([])

#bubble sort because I can't figure out python sort functions for 2d
for i in range(len(fpi_predict_list)-1, 0, -1):
    for j in range(i):
        if(fpi_predict_list[j][1] < fpi_predict_list[j+1][1]):
            temp = fpi_predict_list[j]
            fpi_predict_list[j] = fpi_predict_list[j+1]
            fpi_predict_list[j+1] = temp

#print out sorted list
for k in range(len(fpi_predict_list)):
    print(f"{fpi_predict_list[k][0]}: {fpi_predict_list[k][1]}")
"""

#part 2, chance of every team to make a bowl game and playoff

#first is the bowl game
"""
log = LogisticRegression()
log.fit(x2_2025_train, y1_2025_train)
y_predict = log.predict_proba(x2_2025_test)
y_predict = y_predict[:,1]
y_predict = np.round(y_predict, decimals=4)
bowl_predict_list = [[]]
print()
print("Bowl Game Chance (2025) (Sorted)")
for k in range(len(x)):
    bowl_predict_list.append([csvFile['Teams'].iloc[k], round(y_predict[k]*100,2)])

#removing the first row which is null
bowl_predict_list.remove([])

#bubble sort because I can't figure out python sort functions for 2d
for i in range(len(bowl_predict_list)-1, 0, -1):
    for j in range(i):
        if(bowl_predict_list[j][1] < bowl_predict_list[j+1][1]):
            temp = bowl_predict_list[j]
            bowl_predict_list[j] = bowl_predict_list[j+1]
            bowl_predict_list[j+1] = temp

#print out sorted list
for k in range(len(bowl_predict_list)):
    print(f"{bowl_predict_list[k][0]}: {bowl_predict_list[k][1]}%")
#"""

#now is the playoff chances
"""
log = LogisticRegression()
log.fit(x2_2025_train, y2_2025_train)
y_predict = log.predict_proba(x2_2025_test)
y_predict = y_predict[:,1]
y_predict = np.round(y_predict, decimals=4)
playoff_predict_list = [[]]
print()
print("Playoffs Chance (2025) (Sorted)")
for k in range(len(x)):
    playoff_predict_list.append([csvFile['Teams'].iloc[k], round(y_predict[k]*100,2)])

#removing the first row which is null
playoff_predict_list.remove([])

#bubble sort because I can't figure out python sort functions for 2d
for i in range(len(playoff_predict_list)-1, 0, -1):
    for j in range(i):
        if(playoff_predict_list[j][1] < playoff_predict_list[j+1][1]):
            temp = playoff_predict_list[j]
            playoff_predict_list[j] = playoff_predict_list[j+1]
            playoff_predict_list[j+1] = temp

#print out sorted list
for k in range(len(playoff_predict_list)):
    print(f"{playoff_predict_list[k][0]}: {playoff_predict_list[k][1]}%")
"""
