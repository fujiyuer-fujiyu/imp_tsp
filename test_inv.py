from scipy.io import wavfile
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

#(rate,data)=wavfile.read("./tmp.wav")
#(rate,data)=wavfile.read("./20180228/ref/rec_tsp.wav")
(rate,data)=wavfile.read("tsp_out.16.wav")
#data=sig.lfilter(np.array([0.5,0,0,0.3]), np.ones(1), data)
i_length=data.shape[0]
###
#data2 = np.zeros(data.shape, data.dtype)
#length=data2.shape[0]
#print data.dtype
#print length
print "data.shape " + str(data.shape)
#print "Recoderd wavfile's shape {0} {1}".format(data.shape[0],data.shape[1])
#for i in range(0, length):
#    data2[i] = data[length -i -1]
#wavfile.write("./invtsp.wav", rate, data2)

(rate,data2)=wavfile.read("./itsp_out.wav")
print "data2.shape"+str(data2.shape)
length = data2.shape[0]

print "data" + str(data[977:1000][4])

## synch addition
data_s = np.zeros(length)
for s in range(0,int(i_length/length)):
    print str(s)+"th addition.. "
    for i in range(0,length):
        data_s[i] += data[i+s*length]
        #data_s[i] += data[i+s*length][0]
    

data3 = np.zeros(data.shape, data.dtype)
#data3=sig.fftconvolve(data,data2,mode='full')
data3=sig.fftconvolve(data_s,data2,mode='full')
print data3.shape
print data3.dtype
wavfile.write("./convoled.wav",rate,data3)
plt.figure()
plt.title("Convolved")
plt.plot(data3)

#data4_f=np.fft.fft(data,length)*np.fft.fft(data2,length)
data4_f=np.fft.fft(data_s,length)*np.fft.fft(data2,length)
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
plt.title("fft-Convolved")
plt.plot(data4_i)
#plt.figure()
#plt.plot(data4.imag)
#plt.figure()
#plt.plot(np.abs(data4))
plt.show()
