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
        return  self.challenge

    def Response(self):
        self.result = dot(self.parameters,self.feature)
        self.sig = sign(self.result)
        self.response = 0.5 *self.sig + 0.5
        return self.response

if __name__ == '__main__':
    numbits = 192
    numsample = 3000
    challenge = random.randint(0, 2, [numbits, numsample])

    FirAPUF = ArbiterPUF(numbits/3,numsample,challenge[0:numbits/3])
    SecAPUF = ArbiterPUF(numbits/3,numsample,challenge[numbits/3:(2*numbits)/3])


    FirAPUF.Parameters()
    SecAPUF.Parameters()

    FirAPUF.Challenge()
    SecAPUF.Challenge()

    a = FirAPUF.Response()
    b = SecAPUF.Response()

    challengebit = 1 - a * b



    challenge[(5*numbits)/6] = challengebit
    ThirAPUF = ArbiterPUF(numbits / 3,numsample,challenge[(2*numbits)/3:numbits])
    ThirAPUF.Parameters()
    ThirAPUF.Challenge()
    c = ThirAPUF.Response()




