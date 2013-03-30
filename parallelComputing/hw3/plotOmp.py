from pylab import *
import matplotlib.pyplot as plt

def plotOmp(scan, sscan, numThreads, title):
    x = range(2, numThreads+1)
    figure()
    plt.plot( x, scan, 'r^--', x, sscan, 'bs-' )
    plt.xlabel('Threads')
    plt.ylabel('Time(s)')
    plt.legend(['scan','sscan'])
    plt.title(title)
    plt.savefig(title+'.png')
    
