from scipy.io import wavfile
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

(rate,data)=wavfile.read("./tsp_out.wav")
data=sig.lfilter(np.array([0.5,0,0,0.3]), np.ones(1), data)
length=data.shape[0]
###
data2 = np.zeros(data.shape, data.dtype)
length=data2.shape[0]
print data.dtype
print length
print data.shape
print data2.shape
for i in range(0, length):
    data2[i] = data[length -i -1]
wavfile.write("./invtsp.wav", rate, data2)

#(rate,data2)=wavfile.read("./itsp_out.wav")

data3 = np.zeros(data.shape, data.dtype)
data3=sig.fftconvolve(data,data2,mode='full')
print data3.shape
print data3.dtype
wavfile.write("./convoled.wav",rate,data3)
plt.figure()
plt.plot(data3)

data4_f=np.fft.fft(data,length)*np.fft.fft(data2,length)
data4=np.fft.ifft(data4_f)
print data4.shape
print data4.dtype
data4_i = np.zeros(length, dtype=np.int16)
data4_i = 9000*data4.real/np.max(data4.real)
data4_i = data4_i.astype(np.int16)
print data4_i.dtype
#wavfile.write("./fft-convoled.wav",rate,np.abs(data4))
wavfile.write("./fft-convoled.wav",rate,data4_i)

plt.figure()
#plt.plot(data4.real)
#plt.figure()
plt.plot(data4_i)
#plt.figure()
#plt.plot(data4.imag)
#plt.figure()
#plt.plot(np.abs(data4))
plt.show()
