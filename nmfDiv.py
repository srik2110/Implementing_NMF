import numpy as np
#print 'in function'
eps = 1e-8
def nmfDiv(V, W, H, iter, nB, Divg):
	n, m = np.shape(V)
	for i in range(iter):
		Wcolsum = W.sum(axis = 0)#col sum
		#print Wcolsum
		Wc = Wcolsum.repeat(m)
		#print Wc
		WcMat = Wc.reshape(nB, m)
		#print WcMat
		H = H*((W.T.dot(V/(W.dot(H))))/WcMat)

		Hrowsum = H.sum(axis = 1)
		Hr =  Hrowsum.repeat(n)
		#print Hr
		HrMat = Hr.reshape(n, nB)
		#print HrMat
		W = W*(((V/W.dot(H)).dot(H.T))/HrMat)
		est = W.dot(H)
		#z = np.where(est == 0)[0]
		V = V+eps
		est = est+eps
		D = np.sum(V*np.log(V/est)-V+est)
		Divg.append(D)
                

	#print Divg
	return [W, H, Divg]
