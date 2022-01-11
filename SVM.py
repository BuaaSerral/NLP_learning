from sklearn import svm
import numpy as np
import pylab as pl

#generate data
np.random.seed(5)
x = np.r_[
    np.random.randn(20,2) + [2,2],
    np.random.randn(20,2) - [2,2]
]
y = [0] * 20 + [1] * 20

#define classifier
clf = svm.SVC(kernel= 'linear')
#train
clf.fit(x,y)

'''
w[0]x[0] + w[1]x[1] + b = 0
k = -w[0]/w[1]
b = b/w[1]
'''
W = clf.coef_[0]
B = clf.intercept_[0]
k = -W[0]/W[1]
b = -B/W[1]

#generate target line
xx = np.linspace(-5,5)
yy = k * xx + b

#and 2 support vector lines
yy_down = k * xx + b - 1 / W[1]
yy_up = k * xx +b + 1/W[1]

#draw
pl.plot(xx,yy,'k--')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

pl.scatter(x[:,0],x[:,1],c = 'purple',cmap=pl.cm.Paired)
pl.show()