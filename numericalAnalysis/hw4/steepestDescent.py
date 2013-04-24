import numpy as np

t = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [6.8, 3.0, 1.5, 0.75, 0.48, 0.25, 0.20, 0.15]

arrT = np.asarray(t)
arrY = np.asarray(y)

def errF(x1, x2):
    return  np.sum( (x1*np.exp(x2*arrT) - arrY)**2 )


def getGrad(x1, x2):
    gradX1 = np.sum(2*((x1*np.exp(x2*arrT)) - arrY)*np.exp(x2*arrT))
    gradX2 = np.sum((2*((x1*np.exp(x2*arrT)) - arrY)*np.exp(x2*arrT)*x1)*arrT) 
    return np.asarray([gradX1, gradX2])


def lineSearchDescVal(alpha, x1, x2):
    grad = getGrad(x1, x2)
    diffX1 = x1 - alpha*grad[0]
    diffX2 = x2 - alpha*grad[1]
    return errF(diffX1, diffX2)


#will work only if unimodal in [start, end]
def goldSearchAlpha(start, end, x1, x2, tol):
    tau = (np.sqrt(5) - 1) /2

    alpha1 = start+ (1-tau)*(end - start)
    f1 = lineSearchDescVal(alpha1, x1, x2)

    alpha2 = start+ (tau)*(end - start)
    f2 = lineSearchDescVal(alpha2, x1, x2)

    while end - start> tol:
        if f1 > f2:
            start= alpha1
            alpha1= alpha2
            f1 = f2
            alpha2 = start+ (tau)*(end - start)
            f2 = lineSearchDescVal(alpha2, x1, x2)
        else:
            end = alpha2
            alpha2 = alpha1
            f2 = f1
            alpha1 = start+ (1-tau)*(end - start)
            f1 = lineSearchDescVal(alpha1, x1, x2)
    return start


def findUniModalIntval(x1, x2):
    pt = 4
    p = 0
    end = 0
    fRight = lineSearchDescVal(pt+1, x1, x2)
    fLeft = lineSearchDescVal(pt-1, x1, x2)

    maxIter = 50
    delta = 2
    #print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft
    if fRight > fLeft:
        q = pt+1
        pt = q
        #search in left a point less than ptRight
        while maxIter > 0:
            maxIter -= 1
            pt -= delta
            fLeft = lineSearchDescVal(pt, x1, x2)
            #print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft
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
            fRight = lineSearchDescVal(pt, x1, x2)
            #print 'pt: ', pt, ' fRight: ', fRight, ' fLeft: ', fLeft
            if fLeft < fRight:
                q = pt
                break
    #print 'interval found: ', (p, q)
    return (p, q)   
    

def steepDescent(startX1, startX2, tol):

    x1 = startX1
    x2 = startX2

    gradX1 = -1.0
    gradX2 = -1.0
    alpha = -1.0

    print 'iter','val', 'alpha', 'x1', 'x2', 'gradX1', 'gradX2'
    
    for i in range(500):
        curr = errF(x1, x2)

        print i, curr, alpha, x1, x2, gradX1, gradX2

        grad = getGrad(x1, x2)
        gradX1 = grad[0]
        gradX2 = grad[1]
        (a, b) = findUniModalIntval(x1, x2)
        alpha = goldSearchAlpha(a, b, x1, x2, tol)
        oldX1 = x1
        oldX2 = x2
        x1 = x1 - alpha*gradX1
        x2 = x2 - alpha*gradX2
        new = errF(x1, x2)
        if np.abs(x1-oldX1) < tol and np.abs(x2 - oldX2) < tol:
            print i, curr, alpha, x1, x2, gradX1, gradX2
            return (x1, x2)

    print 'not found steep descent'
    return (0,0)
            

steepDescent(10.0, -1, 0.00000001)

