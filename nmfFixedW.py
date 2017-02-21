import numpy as np
#print 'in function'
eps = 1e-8
def nmfFW(V, W, H, iter, nB, Divg):
	n, m = np.shape(V)
	for i in range(iter):
		Wcolsum = W.sum(axis = 0)
		Wc = Wcolsum.repeat(m)
		WcMat = Wc.reshape(nB, m)
		H = H*(W.T.dot(V/(W.dot(H))))/WcMat

		est = W.dot(H)
		#z = np.where(est == 0)[0]
		V = V+eps
		est = est+eps
		D = np.sum(V*np.log(V/est)-V+est)
		Divg.append(D)

	#print Divg
	return [H, Divg]
