from numpy import *
from lr import *
import copy
import time

class ArbiterPUF(object):

    def __init__(self,numbits,numsample,challenge):
        self.numbits = numbits
        self.numsample = numsample
        self.challenge = challenge
    def Parameters(self):
        self.parameters = random.normal(0, 100, self.numbits)

    def Challenge(self):
        self.feature = [prod(1 - 2 * self.challenge[i:, :], 0) for i in range(self.numbits)]
        return self.feature


    def Response(self):
        self.result = dot(self.parameters,self.feature)
        self.sig = sign(self.result)
        self.response = 0.5 *self.sig + 0.5
        return self.response

class SRAMPUF(object):

    def __init__(self,numbers):
        self.numbers = numbers

    def cell(self):
        self.CellSets = random.randint(0, 2, 2 * self.numbers)
        return  self.CellSets

class AttackForAPUF(object):

    def __init__(self,DataSet,label,numbits,numsample,numIt):
        self.DataSet = DataSet
        self.label = label
        self.numbits = numbits
        self.numIt = numIt
        self.numsample = numsample

    def EncChallenges(self):
        self.feature = [prod(1 - 2 * self.DataSet[i:, :], 0) for i in range(self.numbits)]
        self.data = [['' for x in range(self.numbits)] for y in range(self.numsample)]
        for j in range(self.numsample):
            for i in range(self.numbits):
                self.data[j][i] = self.feature[i][j]

    def TrainData(self):
        self.LRClassifier = logisticRegres()
        classifierArray = self.LRClassifier.train(self.data[:self.numsample / 2], self.label[:self.numsample / 2],numIter=self.numIt)
        classest = self.LRClassifier.classifyArray(self.data[:self.numsample / 2])
        labelMat = mat(self.label[:self.numsample / 2])
        errArr1 = mat(ones(labelMat.shape))
        errSum1 = errArr1[classest != labelMat].sum()
        return errSum1, 1 - (errSum1 / len(self.data[:self.numsample / 2]))

    def TestData(self):
        classestMat = self.LRClassifier.classifyArray(self.data[self.numsample/2:])
        testlabelMat = mat(self.label[self.numsample/2:])
        errArr2 = mat(ones(testlabelMat.shape))
        errSum2 = errArr2[classestMat != testlabelMat].sum()
        return errSum2,1-(errSum2/(self.numsample/2))

if __name__ == '__main__':
    Numbits = 128
    Numsample = 3000
    NumIter = 500
    challenges = random.randint(0, 2, [Numbits,Numsample])
    sramunit = SRAMPUF(Numbits)
    sramoutput = sramunit.cell()


    def MUX(num,numsample,inputdata,inputsig):
        output = [[0 for x in range(numsample)] for y in range(num)]
        for j in range(numsample):
            for i in range(num):
                if (inputsig[i][j] == 1):
                    output[i][j] = inputdata[2 * i - 1]
                else:
                    output[i][j] = inputdata[2 * i]
        return output

    challengeForAPUF = MUX(Numbits,Numsample,sramoutput,challenges)


    ArbPUF = ArbiterPUF(Numbits,Numsample,array(challengeForAPUF))
    ArbPUF.Parameters()
    C = ArbPUF.Challenge()
    R = ArbPUF.Response()

    APUFAttack = AttackForAPUF(challenges, list(R), Numbits, Numsample, NumIter)
    APUFAttack.EncChallenges()
    TrainResult = APUFAttack.TrainData()
    TestResult = APUFAttack.TestData()
    print TrainResult,TestResult




