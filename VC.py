import csv
from scipy import stats
import numpy as np
import math

RawData = open('voice.csv','r')
FormattedData = list(csv.reader(RawData))

TrainingData = []

def ConvertDataIntoFloat(L):
    Width = len(L)
    BlankList = []

    for Data in L[0:(Width-1)]:
        BlankList.append(float(Data))

    BlankList.append(L[Width-1])

    return BlankList


for Data in FormattedData[85:3085]:
    TrainingData.append(ConvertDataIntoFloat(Data))

#Going to calculate MVU Estimates of Natural Parameters of Gaussian Distribution of Mean Frequency for Male and Female separately

MeanFreqMale = []
MeanFreqFemale = []

for Data in TrainingData:

    if Data[20] == 'male':
        MeanFreqMale.append(Data[0])
    else:
        MeanFreqFemale.append(Data[0])

MVUEstimateMUMaleMeanFreq = np.mean(MeanFreqMale)
MVUEstimateVARMaleMeanFreq = np.var(MeanFreqMale)

MVUEstimateMUFemaleMeanFreq = np.mean(MeanFreqFemale)
MVUEstimateVARFemaleMeanFreq = np.var(MeanFreqFemale)

TestingData = []

for Data in FormattedData[1:85]:
    TestingData.append(ConvertDataIntoFloat(Data))

for Data in FormattedData[3085:3169]:
    TestingData.append(ConvertDataIntoFloat(Data))

#Going to perform testing of Bayes Classifier
CorrectCount =0

for Data in TestingData:

    PosteriorMale = stats.norm.pdf(Data[0],MVUEstimateMUMaleMeanFreq,math.sqrt(MVUEstimateVARMaleMeanFreq))
    PosteriorFemale = stats.norm.pdf(Data[0],MVUEstimateMUFemaleMeanFreq,math.sqrt(MVUEstimateVARFemaleMeanFreq))

    PriorMale = (PosteriorMale)/(PosteriorMale + PosteriorFemale)
    PriorFemale = (PosteriorFemale)/(PosteriorMale + PosteriorFemale)

    if PriorMale > PriorFemale and Data[20] == 'male':
        CorrectCount += 1

    elif PriorMale < PriorFemale and Data[20] == 'female':
        CorrectCount += 1

print("The accuracy of our implemented Male/Female Voice Bayes Classifier for single feature is " + str((CorrectCount/(len(TestingData)))))





