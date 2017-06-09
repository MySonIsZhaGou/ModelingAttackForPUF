from numpy import *
from lr import *
import copy
import time

class ArbiterPUF(object):

    def __init__(self,numbits,numsample):
        self.numbits = numbits
        self.numsample = numsample

    def Parameters(self):
        self.parameters = random.normal(0, 100, self.numbits)

    def Challenge(self):
        self.challenges = random.randint(0, 2, [self.numbits, self.numsample])
        self.feature = [prod(1 - 2 * self.challenges[i:, :], 0) for i in range(self.numbits)]
        return  self.challenges

    def Response(self):
        self.result = dot(self.parameters,self.feature)
        self.sig = sign(self.result)
        self.response = 0.5 *self.sig + 0.5
        return self.response

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


class ProStroPUF(object):

    def __init__(self,numbits,numsample):
        self.numbits = numbits
        self.numsample = numsample

    def SegmentParameters(self):
        self.parameters = random.normal(0, 100, 8 * self.numbits)
        return self.parameters

    def Challenge(self):
        challenges = random.randint(0, 2, [2 * self.numbits, self.numsample])
        self.features = 1 - 2 * challenges
        return challenges
        #the challenge should be self.challenge ,it's better
        #the return value is not always a good choice, and it's better to invoke property

    def Feature(self):
        self.feature = []
        for i in range(self.numbits):
            self.feature.append([])
        for j in range(self.numbits):
            self.feature[0].append((1 + self.features[2 * j]) * (1 + self.features[2 * j + 1]))
            self.feature[1].append((1 - self.features[2 * j]) * (1 + self.features[2 * j + 1]))
            self.feature[2].append((1 + self.features[2 * j]) * (1 - self.features[2 * j + 1]))
            self.feature[3].append((1 - self.features[2 * j]) * (1 - self.features[2 * j + 1]))
        for k in range(self.numbits)[::-1]:
            self.feature[4].append((1 + self.features[2 * k]) * (1 + self.features[2 * k + 1]))
            self.feature[5].append((1 + self.features[2 * k]) * (1 - self.features[2 * k + 1]))
            self.feature[6].append((1 - self.features[2 * k]) * (1 + self.features[2 * k + 1]))
            self.feature[7].append((1 - self.features[2 * k]) * (1 - self.features[2 * k + 1]))
        for m in range(self.numbits):
            self.feature[m] = array(self.feature[m])
        return self.feature

    def Response(self):
        Delta = []
        for i in range(self.numbits):
            Delta.append(dot(self.parameters[i * self.numbits : (i + 1) * self.numbits] , self.feature[i]))
        DeltaTop = 0.25 * (Delta[0] + Delta[1] + Delta[2] + Delta[3])
        DeltaBottom = 0.25 * (Delta[4] + Delta[5] + Delta[6] + Delta[7])
        DeltaDifference = DeltaTop - DeltaBottom
        SigOutput = sign(DeltaDifference)
        response = 0.5 * (1 - SigOutput)
        return response

class AttackForPPUF(object):

    def __init__(self,DataSet,label,numbits,numsample,numIt):
        self.DataSet = DataSet
        self.label = label
        self.numbits = numbits
        self.numIt = numIt
        self.numsample = numsample
    def EncChallenge(self):
        self.data = 1 - 2 * self.DataSet

    def Feature_FOR_struct1(self):
        self.DataFeature = []
        for i in range(self.numbits):
            self.DataFeature.append([])
        for j in range(self.numbits):
            self.DataFeature[0].append((1 + self.data[2 * j]) * (1 + self.data[2 * j + 1]))
            self.DataFeature[1].append((1 - self.data[2 * j]) * (1 + self.data[2 * j + 1]))
            self.DataFeature[2].append((1 + self.data[2 * j]) * (1 - self.data[2 * j + 1]))
            self.DataFeature[3].append((1 - self.data[2 * j]) * (1 - self.data[2 * j + 1]))
        for k in range(self.numbits)[::-1]:
            self.DataFeature[4].append((1 + self.data[2 * k]) * (1 + self.data[2 * k + 1]))
            self.DataFeature[5].append((1 + self.data[2 * k]) * (1 - self.data[2 * k + 1]))
            self.DataFeature[6].append((1 - self.data[2 * k]) * (1 + self.data[2 * k + 1]))
            self.DataFeature[7].append((1 - self.data[2 * k]) * (1 - self.data[2 * k + 1]))
        for m in range(self.numbits):
            self.DataFeature[m] = array(self.DataFeature[m])

        self.dataset = [['' for x in range(8 * self.numbits)] for y in range(self.numsample)]
        for j in range(self.numsample):
            for n in range(8):
                for i in range(self.numbits):
                    self.dataset[j][self.numbits * n + i] = self.DataFeature[n][i][j]

    def TrainData(self):
        self.LRClassifier = logisticRegres()
        classifierArray = self.LRClassifier.train(self.dataset[:self.numsample/2], self.label[:self.numsample/2], numIter = self.numIt)
        classest = self.LRClassifier.classifyArray(self.dataset[:self.numsample/2])
        labelMat = mat(self.label[:self.numsample/2])
        errArr1 = mat(ones(labelMat.shape))
        errSum1 = errArr1[classest != labelMat].sum()
        return errSum1, 1-(errSum1/len(self.dataset[:self.numsample/2]))

    def TestData(self):
        classestMat = self.LRClassifier.classifyArray(self.dataset[self.numsample/2:])
        testlabelMat = mat(self.label[self.numsample/2:])
        errArr2 = mat(ones(testlabelMat.shape))
        errSum2 = errArr2[classestMat != testlabelMat].sum()
        return errSum2,1-(errSum2/(self.numsample/2))

class TriInvPUF(object):

     def __init__(self,numbits,numsample):
         self.numbits = numbits
         self.numsample = numsample
         self.HalfNumbits = int(self.numbits / 2)

     def DrivingCapability(self):
         self.capability = abs(random.normal(0, 1, self.numbits))

     def Challenge(self):
         self.challenges = random.randint(0, 2, [self.HalfNumbits, self.numsample])
         self.challenges1 = copy.deepcopy(self.challenges)
         self.challenges2 = copy.deepcopy(self.challenges)
         random.shuffle(self.challenges2)
         self.dataset = [[0 for x in range(self.numbits)] for y in range(self.numsample)]
         for j in range(self.numsample):
             for i in range(self.numbits / 2):
                 self.dataset[j][i] = (self.challenges1[i][j])
             for k in range(self.numbits / 2):
                 self.dataset[j][k + self.numbits/2] = (self.challenges2[k][j])
         return self.dataset

     def Response(self):
         self.response = []
         self.moduel1 = dot(self.capability[:self.HalfNumbits], self.challenges1)
         self.moduel2 = dot(self.capability[self.HalfNumbits:], self.challenges2)
         for j in range(self.numsample):
            if self.moduel1[j] > self.moduel2[j]:
                self.response.append(1)
            else:
                self.response.append(0)
         return self.response

class AttackForTPUF(object):

    def __init__(self,DataSet,label,numbits,numsample,numIt):
        self.DataSet = DataSet
        self.label = label
        self.numbits = numbits
        self.numIt = numIt
        self.numsample = numsample

    def TrainData(self):
        self.LRClassifier = logisticRegres()
        classifierArray = self.LRClassifier.train(self.DataSet[:self.numsample / 2], self.label[:self.numsample / 2],numIter=self.numIt)
        classest = self.LRClassifier.classifyArray(self.DataSet[:self.numsample / 2])
        labelMat = mat(self.label[:self.numsample / 2])
        errArr1 = mat(ones(labelMat.shape))
        errSum1 = errArr1[classest != labelMat].sum()
        return errSum1, 1 - (errSum1 / len(self.DataSet[:self.numsample / 2]))

    def TestData(self):
        classestMat = self.LRClassifier.classifyArray(self.DataSet[self.numsample/2:])
        testlabelMat = mat(self.label[self.numsample/2:])
        errArr2 = mat(ones(testlabelMat.shape))
        errSum2 = errArr2[classestMat != testlabelMat].sum()
        return errSum2,1-(errSum2/(self.numsample/2))
1


if __name__ == '__main__':

    NumBits = input("Please enter the numberbits:")
    NumSamples = input("Please enter the number of the samples:")
    print "(",NumSamples/2,"samples for train data and",NumSamples/2,"samples for test data )",'\n',
    NumIter = input("Please enter the number of the interation you prefer(500 is recommended):")
    choose = input("Please enter the number of the PUF you want to test:"
                   "\n1:FPGA-based Strong PUF, 2:Tristate inverter array PUF, 3:Arbiter PUF\n")
    print "Please waiting......"

    def execution(choose,NumBits,NumSamples,NumIter):
        if choose == 1:
            PPUFStart = time.time()
            PPUF = ProStroPUF(NumBits,NumSamples)
            PPUF.SegmentParameters()
            challengesSet = PPUF.Challenge()
            PPUF.Feature()
            ResponseSet = PPUF.Response()

            PPUFAttack = AttackForPPUF(challengesSet,list(ResponseSet),NumBits,NumSamples,NumIter)
            PPUFAttack.EncChallenge()
            PPUFAttack.Feature_FOR_struct1()
            TrainResult = PPUFAttack.TrainData()
            TestResult = PPUFAttack.TestData()
            PPUFEnd = time.time()
            return TrainResult,TestResult,str(PPUFEnd - PPUFStart)

        elif choose == 2:
            TPUFStart = time.time()
            TPUF= TriInvPUF(NumBits,NumSamples)
            TPUF.DrivingCapability()
            TPUF.Challenge()
            TPUF.Response()

            TPUFAttack = AttackForTPUF(TPUF.Challenge(),TPUF.Response(),NumBits,NumSamples,NumIter)
            TrainResult = TPUFAttack.TrainData()
            TestResult = TPUFAttack.TestData()
            TPUFEnd = time.time()
            return TrainResult, TestResult, str(TPUFEnd - TPUFStart)

        elif choose == 3:
            APUFStart = time.time()
            APUF = ArbiterPUF(NumBits,NumSamples)
            APUF.Parameters()
            APUF.Challenge()
            APUF.Response()

            APUFAttack = AttackForAPUF(APUF.Challenge(),list(APUF.Response()),NumBits,NumSamples,NumIter)
            APUFAttack.EncChallenges()
            TrainResult = APUFAttack.TrainData()
            TestResult = APUFAttack.TestData()
            APUFEnd = time.time()
            return TrainResult, TestResult, str(APUFEnd - APUFStart)

    executionResult = execution(choose,NumBits,NumSamples,NumIter)
    print "The count of train pridict error are:",int(executionResult[0][0]),'\n',\
          "The rate of train pridict error is:",str(executionResult[0][1] * 100 )+'%','\n',\
          "The count of test pridict error are:",int(executionResult[1][0]),'\n',\
          "The rate of test pridict error is:",str(executionResult[1][1] * 100 )+'%','\n', \
          "The execution time is", str(executionResult[2])+"s"

