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

    for EachData in TrainingData[1500:3000]:
        b=0
        b=math.exp(-(Theta0+(Theta1*EachData[0])))
        Del_Theta0_Female = Del_Theta0_Female - (1/(1+b))
        Del_Theta1_Female = Del_Theta1_Female - (EachData[0]/(1+b))

    Del_Theta0 = Del_Theta0_Male + Del_Theta0_Female
    Del_Theta1 = Del_Theta1_Male + Del_Theta1_Female

    Theta0 = Theta0 + (alpha * Del_Theta0)
    Theta1 = Theta1 + (alpha * Del_Theta1)

    for EachData in TrainingData[0:1500]:
        c=math.exp(-(Theta0+(Theta1*EachData[0])))
        LF = LF + math.log(1/(1+c))

    for EachData in TrainingData[1500:3000]:
        d=math.exp(-(Theta0+(Theta1*EachData[0])))
        LF = LF + math.log(1 - (1/(1+d)))

    print("The value of Likelihood function at iteration number {} is {}".format(iterations,LF))

    iterations += 1

#testing
CorrectCount = 0
for EachData in TestingData[0:168]:
    e=math.exp(-(Theta0+(Theta1*EachData[0])))
    PriorProb= (1/(1+e))
    if PriorProb > 0.5 and EachData[1] == 'male':
        CorrectCount += 1
    elif PriorProb < 0.5 and EachData[1] == 'female':
        CorrectCount += 1
print('Value of count ='+str(CorrectCount))




