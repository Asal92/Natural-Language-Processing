import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
np.set_printoptions(threshold=np.inf)

WINDOW_SIZE = 400 # 25ms, 400 samples
HALF_WINDOW = 200
STEP_SIZE = 160 # 10 ms, 160 samples


def hamming_window():
    w = np.zeros(WINDOW_SIZE)
    for n in range(WINDOW_SIZE):
        w[n] = 0.54 - 0.46 * np.cos((2 * np.pi * n) / WINDOW_SIZE)
    return w


def normalization(input_array):
    # normalizing and preparing data for visualization
    # (x - min(x)) * (smax - smin) / (max(x) - min(x)) + smin
    min_num = np.min(input_array)
    max_num = np.max(input_array)

    normalized_spectro = ((input_array - min_num) * 255) / (max_num - min_num)
    return normalized_spectro


def data_windows(file_name):
    # 1channel, 16 bit sample, 16kHz sample rate(reads 16k sample per second)

    spectro_array = np.empty((0,HALF_WINDOW))
    window_num = 0

    wf = hamming_window()
    sample_rate, samples = wavfile.read(file_name)
    file_length = len(samples) / sample_rate
    print(f'sample rate is: {sample_rate}')
    print(f'length of file is: {file_length} seconds')
    print(f'length of samples is: {len(samples)}')

    read_indx = 0
    while True:
        '''
        # 1. read the data from n = 0
        # >> if there isn't full window size data, break the loop
        # 2. multiply by hamming func
        # 3. fft
        # 4. magnitude of fft 
        # 5. 10*log10 magnitudes 
        # 6. save it to an array for visualization
        # 7. n + step size (160)
        # 8. repeat from step 1
        '''
        sn = samples[read_indx:read_indx+WINDOW_SIZE] # 1
        if len(sn)<WINDOW_SIZE: # ignoring the window if it has less than 400 samples
            break
        yn = np.multiply(wf, sn) # 2
        fft = np.fft.fft(yn) # 3
        fft_mag = np.zeros(np.shape(fft))
        for n in range(len(fft)):
            fft_mag[n] = np.sqrt((np.real(fft[n]) ** 2) + (np.imag(fft[n]) ** 2)) # 4
        fft_log = 10 * np.log10(fft_mag[1:HALF_WINDOW+1]) # 5  1:201 OR 1:200?
        spectro_array = np.vstack((spectro_array, fft_log)) # 6
        read_indx += STEP_SIZE # 6
        window_num += 1 # to calculate how many window bins we have for visualization

    normalized_spectro = np.abs(normalization(spectro_array)) # normalizing magnitudes to be between 0-255
    #print(len(normalized_spectro))

    # visualization
    bins = np.linspace(0,window_num, num=window_num)
    freq = np.linspace(0,HALF_WINDOW, num=HALF_WINDOW)
    plt.pcolormesh(bins,freq, normalized_spectro.transpose(), cmap="gist_yarg")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Window bin')
    plt.show()


if __name__ == '__main__':
    # given wav file name, display a portion of spectogram of the file
    fname = sys.argv[1]
    data_windows(fname)









