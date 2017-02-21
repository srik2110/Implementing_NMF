import os
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from nmfDiv import nmfDiv
from nmfFixedW import nmfFW

#taken as a empty list as this seems to be faster than numpy array
#IMatrix = []
#i = 0
#path = '/home/srik/MLSP/images/Train'
#takes each file randomly
#for filename in os.listdir(path):
	#join is to used to join the path to the file name
#	img = io.imread(os.path.join(path, filename))
	#print np.shape(img)
        #print filename
#	if img is not None:
		#ravel to change into 1D
#     		imgv = img.ravel()
#		print len(imgv)
#		IMatrix.append(imgv)
#		print(len(IMatrix[i]))
#	        i = i+1
#V = np.asarray(IMatrix)
#print np.shape(V)
#V = V.T
#print np.shape(V)
V = np.empty((10000, 0))
#i = 0
path = '/home/srik/MLSP/images/Train'
for file in os.listdir(path):
	img = io.imread(os.path.join(path, file))
#        print file
#	print np.shape(img)
	imgv = img.ravel()
#	print np.shape(imgv)
	V = np.append(V, imgv[np.newaxis, :].T, axis = 1)

#Need to find basis and weight matrices using NMF
nB = 20
#say V is of size n by m
n, m = np.shape(V)
W = np.random.rand(n, nB)
H = np.random.rand(nB, m)
#print sum(W<0)
#print sum(H<0)
Divg = []
#NMF algorithm
iter = 100
#divergence
#Divg = np.empty()
#print 'here'
[W, H, Divg] = nmfDiv(V, W, H, iter, nB, Divg) 
est = W.dot(H)
#checking one image
img = est[:, 3]
img = np.reshape(img, (100, 100))
plt.gray()
plt.imshow(img)
plt.show()

#try with test images and fixing the basis
test = io.imread('/home/srik/MLSP/images/Test/test_12_img1.gif')
test = test.ravel()
test = test[:, np.newaxis]
#print np.shape(test)
H = np.random.rand(nB, 1)
#print H
Divg = []
[H, Divg] = nmfFW(test, W, H, iter, nB, Divg)
#print H
img = W.dot(H)
img = np.reshape(img, (100, 100))
plt.gray()
plt.imshow(img)
plt.show()
#print np.shape(W)
