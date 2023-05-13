
# coding: utf-8

# In[21]:


import numpy
import scipy.io
import math
import geneNewData
from math import sqrt
from math import pi
from math import exp

def main():
    myID='7088'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    pass
myID='7088'
geneNewData.geneData(myID)
Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
train0 = Numpyfile0.get('target_img')
train1 = Numpyfile1.get('target_img')
test0 = Numpyfile2.get('target_img')
test1 = Numpyfile3.get('target_img')

# Task 1
def features(meanArr0, meanArr1, stdArr0, stdArr1):
    for image in train0:
        meanArr0.append(image.mean())
        stdArr0.append(image.std())
    for image in train1:
        meanArr1.append(image.mean())
        stdArr1.append(image.std())
    #return [meanArr0, meanArr1, stdArr0, stdArr1]
    
# Task 2
def paramaters(meanArr0, meanArr1, stdArr0, stdArr1):
    # feature 1 => mean of brightness => meanArr
    # feature 2 => std of brightness => stdArr
    meanF1Zero = meanArr0.mean()
    meanF2Zero = stdArr0.mean()
    meanF1One = meanArr1.mean()
    meanF2One = stdArr1.mean()
    stdF1Zero =  meanArr0.std()
    stdF2Zero =  stdArr0.std()
    stdF1One = meanArr1.std()
    stdF2One = stdArr1.std()
    return {'meanF1Zero' : meanF1Zero, 'meanF2Zero': meanF2Zero, 'meanF1One': meanF1One, 'meanF2One': meanF2One,
           'stdF1Zero':stdF1Zero, 'stdF2Zero':stdF2Zero, 'stdF1One': stdF1One, 'stdF2One':stdF2One}
    
# Task 3
def featuresTest(testMeanArr0, testMeanArr1, TestStdArr0, TestStdArr1):
    for image in test0:
        testMeanArr0.append(image.mean())
        TestStdArr0.append(image.std())
    for image in test1:
        testMeanArr1.append(image.mean())
        TestStdArr1.append(image.std())

def naive(f, mean, std):
    exponent = exp(-((f-mean)**2 / (2 * std**2 )))
    return (1 / (sqrt(2 * pi) *  std)) * exponent
    
    
def classify(length, testMeanArr0, TestStdArr0, FS):
    classArr = []
    for i in range(length):
        #for zero
        PF1 = naive(testMeanArr0[i], Fs['meanF1Zero'], Fs['stdF1Zero'])
        PF2 = naive(TestStdArr0[i], Fs['meanF2Zero'], Fs['stdF2Zero'])
        zeroProduct = PF1 * PF2
        
        #for one
        PF1 = naive(testMeanArr0[i], Fs['meanF1One'], Fs['stdF1One'])
        PF2 = naive(TestStdArr0[i], Fs['meanF2One'], Fs['stdF2One'])
        oneProduct = PF1 * PF2
        if(zeroProduct >= oneProduct):
            classArr.append(0)
        else:
            classArr.append(1)
    return classArr

if __name__ == '__main__':
    #main()
    
    # for mean for 0 train images => 5000 value
    meanArr0 =[]
    # for mean for 1 train images => 5000 value
    meanArr1 =[]
    # for std for 0 train images => 5000 value
    stdArr0 =[]
    # for std for 1 train images => 5000 value
    stdArr1 =[]
    # task 1
    features(meanArr0, meanArr1, stdArr0, stdArr1)
    # convert arrays to numPy
    meanArr0 = numpy.array(meanArr0)
    meanArr1 = numpy.array(meanArr1)
    stdArr0 = numpy.array(stdArr0)
    stdArr1 = numpy.array(stdArr1)
    # task 2
    Fs = paramaters(meanArr0, meanArr1, stdArr0, stdArr1)
    
    # task 3
    # for mean for 0 test images => 980 value
    testMeanArr0 =[]
    # for mean for 1 test images => 980 value
    testMeanArr1 =[]
    # for std for 0 test images => 1135 value
    TestStdArr0 =[]
    # for std for 1 test images => 1135 value
    TestStdArr1 =[]
    featuresTest(testMeanArr0, testMeanArr1, TestStdArr0, TestStdArr1)

    #zero test class
    zeroClass = classify(len(testMeanArr0), testMeanArr0, TestStdArr0, Fs)
    
    #zero test class
    oneClass = classify(len(testMeanArr1), testMeanArr1, TestStdArr1, Fs)
    
    # task 4
    Accuracy_for_digit0testset = zeroClass.count(0) / 980
    Accuracy_for_digit1testset = oneClass.count(1) / 1135
    print(Fs['meanF1Zero'], Fs['stdF1Zero']**2, Fs['meanF2Zero'], Fs['stdF2Zero']**2,
           Fs['meanF1One'], Fs['stdF1One']**2, Fs['meanF2One'], Fs['stdF2One']**2,
           Accuracy_for_digit0testset, Accuracy_for_digit1testset)

