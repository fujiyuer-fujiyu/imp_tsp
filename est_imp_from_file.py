from scipy.io import wavfile
from scipy import signal as sig
import numpy as np
import matplotlib.pyplot as plt

#input file (observed signal)
(rate,data)=wavfile.read("tsp_out.16.wav")
#(rate,data)=wavfile.read("./20180228/ref/ec_tsp.wav")

i_length=data.shape[0]
print "lenght of Observed signal and number of channels is {0}".format(str(data.shape))

#inverse of TSP signal
(rate,data2)=wavfile.read("./itsp_out.wav")
print "length of Original TSP signal is {}".format(str(data2.shape))
length = data2.shape[0]

#number of chanel
if len(data.shape) > 1:
    nc = data.shape[1]
else :
    nc = 1
print "Number of channel of Observed signal is {}".format(nc)

## synchronous addition
if nc > 1:
    data_s = np.zeros((length,nc))
else:
    data_s = np.zeros(length)

print data_s.shape
for s in range(0,int(i_length/length)):
    print str(s)+"th addition.. "
    for i in range(0,length):
        if nc > 1:
            data_s[i][nc] += data[i+s*length][nc]
        else:
            data_s[i] += data[i+s*length]

# calc inpulse response
data4_f=np.fft.fft(data_s,length)*np.fft.fft(data2,length)
data4=np.fft.ifft(data4_f)
data4_i = np.zeros(length, dtype=np.int16)

# normalize and convert to integer
data4_i = 9000*data4.real/np.max(data4.real)
data4_i = data4_i.astype(np.int16)
wavfile.write("./IR-from-FFT.wav",rate,data4_i)

plt.figure()
plt.title("IR: estimated from FFT")
plt.plot(data4_i)
plt.show()
