from numpy import *
from lr import *
import copy
import time

class ArbArbPUF(object):

    def __init__(self,numbits,numsample):
        self.numbits = numbits
        self.numsample = numsample

    def FirstPara(self):
        self.firparameters = random.normal(0, 100, self.numbits * self.numbits)

    def Challenge(self):
        self.challenges = random.randint(0, 2, [self.numbits, self.numsample])
        self.feature = [prod(1 - 2 * self.challenges[i:, :], 0) for i in range(self.numbits)]
        return self.challenges

    def FirstResponse(self):
        self.firresult = zeros([self.numbits,self.numsample])
        self.sig = zeros([self.numbits, self.numsample])
        self.response = zeros([self.numbits, self.numsample])
        for i in range(self.numbits):
            self.firresult[i] = (dot(self.firparameters[i*self.numbits:((i+1)*self.numbits)],self.feature))
            self.sig[i] = (sign(self.firresult[i]))
            self.response[i] = (0.5 *self.sig[i] + 0.5)
        return self.response

    def SecondPara(self):
        self.secparameters = random.normal(0, 100, self.numbits)

    def SecondResponse(self):
        self.secresult = dot(self.secparameters,[prod(1 - 2 * self.response[i:, :], 0) for i in range(self.numbits)])
        self.secsig = sign(self.secresult)
        self.secresponse = 0.5 *self.secsig + 0.5
        return self.secresponse

class AttackForAAPUF(object):

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
    a = ArbArbPUF(8,16)
    a.FirstPara()
    a.Challenge()
    a.FirstResponse()
    a.SecondPara()
    a.SecondResponse()
    print a.Challenge()
    AAPUFAttack = AttackForAAPUF(a.challenges, list(a.secresponse), 128, 3000, 500)
    AAPUFAttack.EncChallenges()
    TrainResult = AAPUFAttack.TrainData()
    TestResult = AAPUFAttack.TestData()
    print TrainResult,TestResult