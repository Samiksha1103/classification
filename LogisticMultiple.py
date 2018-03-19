import csv
import math
import random

with open("voice.csv","r") as RawData:
    FormattedData = list(csv.reader(RawData))

NecessaryData = []

for EachData in FormattedData[1:]:
    BlankList = []
    BlankList.append(float(EachData[0]))
    BlankList.append(EachData[20])
    NecessaryData.append(BlankList)

TrainingData = NecessaryData[85:3085]
TestingData = []

for EachData in NecessaryData[0:85]:
    TestingData.append(EachData)

for EachData in NecessaryData[3085:3169]:
    TestingData.append(EachData)

#Theta0 = random.randint(0,255)
#Theta1 = random.randint(0,255)
Theta0 = 2
Theta1 = 1
alpha = 0.001


for iterations in range(0,40000):
    Del_Theta0_Male = 0
    Del_Theta1_Male = 0
    Del_Theta0_Female = 0
    Del_Theta1_Female = 0
    Del_Theta0 = 0
    Del_Theta1 = 0
    LF = 0

    for EachData in TrainingData[0:1500]:
        Del_Theta0_Male = Del_Theta0_Male + (1/(1+math.exp(Theta0 + (Theta1*EachData[0]))))
        Del_Theta1_Male = Del_Theta1_Male + (EachData[0]/(1+math.exp(Theta0 + (Theta1*EachData[0]))))
