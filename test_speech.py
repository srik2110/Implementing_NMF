import numpy as np
import wave
import matplotlib.pyplot as plt 
import pylab
from scipy.io import wavfile
from scipy import signal
from nmfDiv import nmfDiv
from nmfFixedW import nmfFW


pylab.close('all')
#w1 = wave.open('/home/srik/MLSP/speechFiles/clean.wav')
#w2 = wave.open('/home/srik/MLSP/speechFiles/noise.wav')
fs, w1 = wavfile.read('/home/srik/MLSP/speechFiles/clean.wav')
fs2, w2 = wavfile.read('/home/srik/MLSP/speechFiles/noise.wav')
fs3, w3 = wavfile.read('/home/srik/MLSP/speechFiles/noisy.wav')

tWindow = 25e-3#Window length in time
NWindow = tWindow*fs#no. of elements in window
NFFT = 400#512 not working - says should be same as segment length!!
window = np.hamming(NWindow)#window type
NOverlap = 0.6*NWindow#60% overlap

#for clean signal
ps, f, t, plot = pylab.specgram(w1, NFFT = NFFT, Fs = fs, window = window, noverlap = NOverlap)
plt.show()
#for noise
psn, fn, tn, plot = pylab.specgram(w2, NFFT = NFFT, Fs = fs, window = window, noverlap = NOverlap)
plt.show()

psny, fny, tny, plot = pylab.specgram(w3, NFFT = NFFT, Fs = fs, window = window, noverlap = NOverlap)
plt.show()


#change the shape to number of frames by freq points
ps = ps.T
V1 = ps
#Need to find basis and weight matrices using NMF for clean speech
nB1 = 50
#say V is of size n by m
n, m = np.shape(V1)
W1 = np.random.rand(n, nB1)
H1 = np.random.rand(nB1, m)
Divg = []
#NMF algorithm
iter = 100
#divergence
#Divg = np.empty()
#print 'here'
[W1, H1, Divg] = nmfDiv(V1, W1, H1, iter, nB1, Divg) 

#Need to find basis and weight matrices using NMF for noise
V2 = psn.T 
nB2 = 50
#say V is of size n by m
n, m = np.shape(V2)
W2 = np.random.rand(n, nB2)
H2 = np.random.rand(nB2, m)
Divg = []
#NMF algorithm
iter = 100
#divergence
#Divg = np.empty()
#print 'here'
[W2, H2, Divg] = nmfDiv(V2, W2, H2, iter, nB2, Divg) 

#find clean speech from noisy speech using the obtained basis
psny = psny.T
V = psny
#Need to find basis and weight matrices using NMF for clean speech
nB = nB1+nB2
#say V is of size n by m
n, m = np.shape(V)
W = np.column_stack((W1, W2))
H = np.random.rand(nB, m)
Divg = []
#NMF algorithm
iter = 100
#divergence
#Divg = np.empty()
#print 'here'
[H, Divg] = nmfFW(V, W, H, iter, nB, Divg) 

#separate H as 50 by cols and other 50 by cols
Hc = H[0:50, :]
Hn = H[50:100, :]
pc = W1.dot(Hc)
pn = W2.dot(Hn)
plt.pcolormesh(t, f, 10*np.log10(pc.T))
plt.xlabel('Time [sec]')
plt.ylabel('Freq [Hz]')
plt.show()

plt.pcolormesh(t, f, 10*np.log10(pn.T))
plt.xlabel('Time [sec]')
plt.ylabel('Freq [Hz]')
plt.show()



#tWindow = 25e-3
#NWindow = tWindow*fs
#window = signal.hamming(NWindow)
#nperseg = 400
#noverlap = 0.6*NWindow
#nfft = 512
#nfft, nperseg are taken by default as NWindow
#mode = 'psd' 
#f, t, Sxx = signal.spectrogram(w1, fs, window, nperseg, noverlap, nfft) 
#plt.pcolormesh(t, f, Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
