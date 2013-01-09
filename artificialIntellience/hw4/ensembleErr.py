#program to compute compound err in independent ensemble learning
def fact(num):

    fact = 1
    for i in range(1, num+1):
        fact = fact * i
    return fact

def combination(n, k):
    return fact(n)/(fact(k)*fact(n-k))

def compoundError(numEnsembles, errProb):
    compoundErr = 0;
    for i in range((numEnsembles/2)+1, numEnsembles+1):
        compoundErr += combination(numEnsembles, i)*\
                         (errProb**i)*((1-errProb)**(numEnsembles-i))
    return compoundErr

print compoundError(5, 0.1)
print compoundError(5, 0.2)
print compoundError(5, 0.4)

print compoundError(10, 0.1)
print compoundError(10, 0.2)
print compoundError(10, 0.4)

print compoundError(20, 0.1)
print compoundError(20, 0.2)
print compoundError(20, 0.4)
        
