import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

def predictFPI(csvFile, xtrain, xtest, ytrain, team="All"):
    #we've already split the data, so no need for train_test_split
    knr = KNeighborsRegressor(n_neighbors=5, p=2, metric='minkowski')
    knr.fit(xtrain,ytrain)
    y_predict = knr.predict(xtest)
    y_predict = np.round(y_predict, decimals=1)
    fpi_predict_list = [[]]
    for k in range(len(xtest)):
        fpi_predict_list.append([csvFile['Teams'].iloc[k], y_predict[k,0], k+1])
        
    #removing the first row which is null
    fpi_predict_list.remove([])

    #bubble sort because I can't figure out python sort functions for 2d
    for i in range(len(fpi_predict_list)-1, 0, -1):
        for j in range(i):
            if(fpi_predict_list[j][1] < fpi_predict_list[j+1][1]):
                temp = fpi_predict_list[j]
                fpi_predict_list[j] = fpi_predict_list[j+1]
                fpi_predict_list[j+1] = temp

    #fix rank column
    for k in range(len(fpi_predict_list)):
        fpi_predict_list[k][2] = k+1
        
    #print out sorted list
    if team == "All":
        for k in range(len(fpi_predict_list)):
            print(f"{fpi_predict_list[k][0]}: {fpi_predict_list[k][1]}")

    else:
        teamN = str.lower(team)
        for k in range(len(fpi_predict_list)):
            if teamN in str.lower(fpi_predict_list[k][0]):
                print(f"{fpi_predict_list[k][0]}: {fpi_predict_list[k][1]} Rank: {fpi_predict_list[k][2]}")

def predictBowlOrPlayoff(csvFile, xtrain, xtest, ytrain, team="All"):
    log = LogisticRegression()
    log.fit(xtrain, ytrain)
    y_predict = log.predict_proba(xtest)
    y_predict = y_predict[:,1]
    y_predict = np.round(y_predict, decimals=4)
    playoff_predict_list = [[]]
    for k in range(len(xtest)):
        playoff_predict_list.append([csvFile['Teams'].iloc[k], round(y_predict[k]*100,2), k])

    #removing the first row which is null
    playoff_predict_list.remove([])

    #bubble sort because I can't figure out python sort functions for 2d
    for i in range(len(playoff_predict_list)-1, 0, -1):
        for j in range(i):
            if(playoff_predict_list[j][1] < playoff_predict_list[j+1][1]):
                temp = playoff_predict_list[j]
                playoff_predict_list[j] = playoff_predict_list[j+1]
                playoff_predict_list[j+1] = temp

    #fix rank column
    for k in range(len(playoff_predict_list)):
        playoff_predict_list[k][2] = k+1
        
    #print out sorted list
    if team == "All":
        for k in range(len(playoff_predict_list)):
            print(f"{playoff_predict_list[k][0]}: {playoff_predict_list[k][1]}%")

    else:
        teamN = str.lower(team)
        for k in range(len(playoff_predict_list)):
            if teamN in str.lower(playoff_predict_list[k][0]):
                print(f"{playoff_predict_list[k][0]}: {playoff_predict_list[k][1]}% Rank: {playoff_predict_list[k][2]}")

def main():
    #insert code here
    csvFile = pd.read_csv("College_Football_Stats.csv")
    #testing dataframes
    xFPI_test = csvFile[['2024 FPI', '2025 Recruiting Class']]
    xBowlPlayoff_test = csvFile[['2024 FPI', '2025 Recruiting Class', '2025 Returning Production']]
    
    #training dataframes
    xFPI_train = csvFile[['2023 FPI', '2024 Recruiting Class']]
    xBowlPlayoff_train = csvFile[['2023 FPI', '2024 Recruiting Class', '2024 Returning Production']]

    #y training
    yFPI_train = csvFile[['2024 FPI']]
    yBowl_train = csvFile[['6+ Wins 2024']]
    yPlayoffs_train = csvFile[['Made Playoffs 2024']]

    #rename columns to prevent model confusion
    xBowlPlayoff_train.columns = ['FPI', 'Recruiting Class', 'Returning Production']
    xBowlPlayoff_test.columns = ['FPI', 'Recruiting Class', 'Returning Production']
    xFPI_train.columns = ['FPI', 'Recruiting Class']
    xFPI_test.columns = ['FPI', 'Recruiting Class']

    #running actual program
    quitting = False
    while quitting == False:
        ans1 = "0"
        while(not(ans1=="1") and not(ans1=="2") and not(ans1=="3")):
            ans1 = input("What would you like to do?\n1. Predict 2025 FPI\n2. Predict Chance to make Bowl Game in 2025\n3. Predict Chance to make Playoffs in 2025\n")
            
        ans2 = input("(Y/N) would you like to specify a specific college team?\n")
        ans2 = str.lower(ans2)
        ans3 = "All"
        if ans2 == "y":
            ans3 = input("Type the college football team here\n")
        if ans1=="1":
            predictFPI(csvFile, xFPI_train, xFPI_test, yFPI_train, ans3)
        if ans1=="2":
            predictBowlOrPlayoff(csvFile, xBowlPlayoff_train, xBowlPlayoff_test, yBowl_train, ans3)
        if ans1=="3":
            predictBowlOrPlayoff(csvFile, xBowlPlayoff_train, xBowlPlayoff_test, yPlayoffs_train, ans3)
            
        ans4 = input("(Y/N) Would you like to quit the program?\n")
        ans4 = str.lower(ans4)
        if ans4=="y":
            quitting = True
        
        



if __name__=="__main__":
    main()
