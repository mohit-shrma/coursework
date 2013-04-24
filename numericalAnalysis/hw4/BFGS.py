import numpy as np


t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [6.8, 3.0, 1.5, 0.75, 0.48, 0.25, 0.20, 0.15]

arrT = np.asarray(t)
arrY = np.asarray(y)

def errF(X):
    x1 = X[0]
    x2 = X[1]
    return  np.sum( (x1*np.exp(x2*arrT) - arrY)**2 )


def getGrad(X):
    x1 = X[0]
    x2 = X[1]
    gradX1 = np.sum(2*((x1*np.exp(x2*arrT)) - arrY)*np.exp(x2*arrT))
    gradX2 = np.sum((2*((x1*np.exp(x2*arrT)) - arrY)*np.exp(x2*arrT)*x1)*arrT) 
    return np.asarray([gradX1, gradX2])


def lineSearchBFGSVal(alpha, X, Sk):
    nextX = X + alpha*Sk
    return errF(nextX)


#will work only if unimodal in [start, end]
def goldSearchBFGSAlpha(start, end, X, Sk, tol):
    tau = (np.sqrt(5) - 1) /2

    alpha1 = start+ (1-tau)*(end - start)   
    f1 = lineSearchBFGSVal(alpha1, X, Sk)

    alpha2 = start+ (tau)*(end - start)
    f2 = lineSearchBFGSVal(alpha2, X, Sk)

    while end - start> tol:
        if f1 > f2:
            start= alpha1
            alpha1= alpha2
            f1 = f2
            alpha2 = start+ (tau)*(end - start)
            f2 = lineSearchBFGSVal(alpha2, X, Sk)
        else:
            end = alpha2
            alpha2 = alpha1
            f2 = f1
            alpha1 = start+ (1-tau)*(end - start)   
            f1 = lineSearchBFGSVal(alpha1, X, Sk)    
    return start



def findUniModalBFGSIntval(X, Sk):
    pt = 0
    p = 0
    q = 0

    fRight = lineSearchBFGSVal(pt + 5, X, Sk)
    fLeft = lineSearchBFGSVal(pt - 5, X, Sk)

    maxIter = 100
    delta = 5
    print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft

    if fRight == fLeft:
        print 'can\'t decide'
        print X
        print Sk
        return (p,q)


    if fRight > fLeft:
        q = pt+1
        pt = q
        #search in left a point less than ptRight
        while maxIter > 0:
            maxIter -= 1
            pt -= delta
            fLeft = lineSearchBFGSVal(pt, X, Sk)
            print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft
            if fLeft > fRight:
                p = pt
                break
    else:
        p = pt-1
        pt = p
        #search in right gr8r than ptLeft
        while maxIter > 0:
            maxIter -= 1
            pt += delta
            fRight = lineSearchBFGSVal(pt, X, Sk)
            print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft
            if fLeft < fRight:
                q = pt
                break

    print 'interval found: ', (p, q)

    return (p, q)   




def BFGS(x1, x2, tol):
    
    
    #hessian approximations
    B = np.eye(2)
    X = np.asarray([x1, x2])
    
    grad = getGrad(X)

    print 'iter','val', 'alpha', 'x1', 'x2', 'gradX1', 'gradX2'

    for i in range(500):
        
        curr = errF(X)
        Sk = np.linalg.lstsq(B, -1*grad)[0]
        print 'B: ', B
        print 'Sk: ', Sk
        print 'grad: ', grad
        #perform line search to solve min<alpha>(e(Xk+alpha*Sk)))
        (p, q) = findUniModalBFGSIntval(X, Sk)
        alpha = goldSearchBFGSAlpha(p, q, X, Sk, tol)
        
        print i, curr, alpha, X[0], X[1], grad[0], grad[1]

        oldX = X
        X = oldX + Sk
        
        if np.abs(np.linalg.norm(oldX) - np.linalg.norm(X)) < tol:
            print i, curr, alpha, X[0], X[1], grad[0], grad[1]
            return (X[0], X[1])
        
        newGrad = getGrad(X)
        Y = newGrad - grad
        Y = Y.reshape(2,1)
        Sk = Sk.reshape(2, 1)
        B = B + (Y*Y.T)/(Y.T*Sk) - ((B*Sk*Sk.T*B)/( np.dot(Sk.T,np.dot(B,Sk)) ) )
        grad = newGrad

    print 'not found BFGS'
    return (0, 0)


BFGS(10.0, -1.0, 0.00000001)
