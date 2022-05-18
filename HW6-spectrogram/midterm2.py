import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.spatial import distance



if __name__ == '__main__':
    window = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    a_x = []
    for x in window:
        a = np.cos(8 * x * np.pi) + np.sin(4 * x * np.pi)
        a_x.append(a)

    b_x = []
    for x in window:
        b = float(3/4) * np.sin((8 * x * np.pi) - (6 * np.pi / 7))
        b_x.append(b)

    c_x = []
    for x in window:
        c = np.sin((8 * x * np.pi) - (3 * np.pi /4)) - np.cos((16 * x * np.pi) + (3 * np.pi / 4))
        c_x.append(c)

    print(a_x)
    print(b_x)
    print(c_x)

    a_fft = np.fft.fft(a_x)
    b_fft = np.fft.fft(b_x)
    c_fft = np.fft.fft(c_x)

    print("a fft is" , a_fft)
    print("b fft is" , b_fft)
    print("c fft is" , c_fft)



    a_mag, b_mag, c_mag = [], [], []
    for n in range(len(a_fft)):
        a_mag.append(np.sqrt((np.real(a_fft[n]) ** 2) + (np.imag(a_fft[n]) ** 2)))
        b_mag.append(np.sqrt((np.real(b_fft[n]) ** 2) + (np.imag(b_fft[n]) ** 2)))
        c_mag.append(np.sqrt((np.real(c_fft[n]) ** 2) + (np.imag(c_fft[n]) ** 2)))

    print(a_mag)
    print("max bin in a is:", np.argmax(a_mag))
    print(b_mag)
    print("max bin in b is:", np.argmax(b_mag))
    print(c_mag)
    print("max bin in c is:", np.argmax(c_mag))


'''
    a_mean = np.mean(a_fft)
    b_mean = np.mean(b_fft)
    c_mean = np.mean(c_fft)
    print("a mean is:", a_mean)

    a_dist, b_dist, c_dist = [], [], []
    for i in range(6):
        a_dist.append(distance.euclidean(a_mag[i], a_mean))
        b_dist.append(distance.euclidean(b_mag[i], b_mean))
        c_dist.append(distance.euclidean(c_mag[i], c_mean))

    print(a_dist)
    print(b_dist)
    print(c_dist)'''


