# Imports
from scipy.io.wavfile import read, write
from scipy.signal import resample, butter, filtfilt
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from pylab import savefig

""" 
Functions 
"""


# Objetive: Save and show a graph using the input arguments
# Input: list with values of X axis, list with values of Y axis, string for title, string for X label
#        string for Y label, String for graph color
def graph(x, y, title, xlabel, ylabel, color):
    plt.figure(title)
    plt.plot(x, y, color, linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    savefig(title)
    plt.show()


# Objetive: Saves and display a Frequency domain graph using the input arguments, also return list of
#           frequencies computed with fourier transform.
# Input: list with values of X axis, list with values of Y axis, string for title, string for X label
#        string for Y label, String for graph color.
# Output: list with amplitudes, list with frequencies
def fourier(data, samplerate, title, xlabel, ylabel, color):
    fouriertransform = fft(data) / len(data)
    fourierfrec = fftfreq(len(data), 1 / samplerate)
    graph(fourierfrec, abs(fouriertransform), title, xlabel, ylabel, color)
    return fouriertransform, fourierfrec


# Objetive: Display and saves a graph for a sinusoidal function
# Input: list with values of X axis, list with values of Y axis, string for title, string for X label
#        string for Y label, String for graph color.
def graphcos(timeinterval, data, title, xlabel, ylabel, color):
    plt.figure(title)
    plt.plot(timeinterval, data, color, linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, 0.005)
    plt.ylim(-1, 1)
    savefig(title)
    plt.show()


"""
Main Section
"""
# Read wav file, retrieving sample rate and data
sampleRateHandel, dataHandel = read("handel.wav")
length = len(dataHandel)  # Number of samples
time = length / sampleRateHandel  # Time in seconds

# Graph Handel signal
handelTimeInterval = np.linspace(0, time, length)
graph(handelTimeInterval, dataHandel, "Handel Signal", "Time [s]", "Amplitude [dB]", "blue")

# Graph fourier transform of Handel Signal
handelFourier, handelFourierFrec = fourier(dataHandel, sampleRateHandel, "Fourier Transform of Handel Signal",
                                           "Frequency [Hz]", "Amplitude [dB]", "blue")
"""
Resample Handel Signal
"""
# Increase number of samples by 10
# Se aumenta la longitud de muestras en diez, para cumplir con el teorema del muestreo
intervalResampledSignal = np.linspace(0, time, length * 10)
# Apply resample() function to get all the missing samples
resampledSignalData = resample(dataHandel, length * 10)
# Compute the new sample rate by dividing N°samples/time
newSampleRate = int(len(resampledSignalData) / time)

"""
Amplitude Modulation
"""
# Creation of the carrier signal
carrierFrequency = 20000
wct = 2 * np.pi * carrierFrequency * intervalResampledSignal
carrierSignal = np.cos(wct)

graphcos(intervalResampledSignal[:600], carrierSignal[:600], "Carrier Signal cos(2πft)",
         "Time [s]", "Amplitude [dB]", "green")

fourier(carrierSignal, newSampleRate, "Fourier Transform of Carrier Signal",
        "Frequency [Hz]", "Amplitude [dB]", "green")

# Generate AM signals using two modulation index
k1 = 1  # Modulation index
k2 = 1.25  # Modulation index
modulatedSignal1 = k1 * resampledSignalData * carrierSignal
modulatedSignal2 = k2 * resampledSignalData * carrierSignal

# Graph the AM signals
graph(intervalResampledSignal, modulatedSignal1, "AM Signal K = 1", "Time [s]", "Amplitude [dB]", "green")
graph(intervalResampledSignal, modulatedSignal2, "AM Signal K = 1,25", "Time [s]", "Amplitude [dB]", "green")
# Fourier transform of the AM signals
fourier(modulatedSignal1, newSampleRate, "Fourier Transform of AM Signal K = 1",
        "Frequency [Hz]", "Amplitude [dB]", "purple")
fourier(modulatedSignal2, newSampleRate, "Fourier Transform of AM Signal K = 1,25",
        "Frequency [Hz]", "Amplitude [dB]", "purple")
# Write the AM signal with K = 1 to a wav file
write("ModulatedHandelAM.wav", newSampleRate, modulatedSignal1.astype(np.int16))

""" 
Demodulation 
"""
# To demodulate is necessary to multiply the AM signal by the carrier signal
signalDemodData = modulatedSignal1 * carrierSignal
signalDemodData2 = modulatedSignal2 * carrierSignal
# Graph the Fourier transform
fourier(signalDemodData, newSampleRate, "Fourier Transform of AM signal K = 1 x carrierSignal", "Frequency [Hz]",
        "Amplitude [dB]", "brown")
fourier(signalDemodData2, newSampleRate, "Fourier Transform of AM signal K = 1,25 x carrierSignal", "Frequency [Hz]",
        "Amplitude [dB]", "brown")

# Apply lowpass filter to recover the original signal
nyquist = 0.5 * newSampleRate
cutoffFrequency = 4096
cutoff = cutoffFrequency / nyquist
# Create a butterworth filter
values = butter(8, cutoff, btype='lowpass')
b2, a2 = values[0], values[1]
# Apply the filter to the AM signals
filteredSignal = filtfilt(b2, a2, signalDemodData)
filteredSignal2 = filtfilt(b2, a2, signalDemodData2)
# Graph the demodulated signals
graph(intervalResampledSignal, filteredSignal, "Demodulated AM signal k = 1", "Time [s]", "Amplitude [dB]", "pink")
graph(intervalResampledSignal, filteredSignal2, "Demodulated AM signal k = 1,25", "Time [s]", "Amplitude [dB]", "pink")
# Writes the demodulated signals back into a wav file
write("DemodulatedHandelk1.wav", newSampleRate, filteredSignal.astype(np.int16))
write("DemodulatedHandelk1,25.wav", newSampleRate, filteredSignal2.astype(np.int16))

"""
Bandwidth
"""
# Find the max frequency in the handel signal
maxIndex = np.argmax(handelFourier)
maxFrecuencym = handelFourierFrec[maxIndex]  # fm
# Bandwidth = (fc + fm) - (fc - fm)
# fc = 20.000 [hz] - Carrier Frecuency
bandWidth = (carrierFrequency + maxFrecuencym) - (carrierFrequency - maxFrecuencym)
print("Bandwidth = ", bandWidth, "[Hz]")

"""
Extra: Adding noise to Handel signal
"""
# To see how noise impacts modulation:
# Create a random noise
noise = np.random.normal(-10000, 10000, length*10)
noiseHandel = []
# Add noise to the Handel signal
for i in range(0, len(resampledSignalData)):
    sumNoiseHandel = noise[i] + resampledSignalData[i]
    noiseHandel.append(sumNoiseHandel)

# graph the signal
graph(intervalResampledSignal, noiseHandel, "Resampled Handel + noise", "Time [s]", "Amplitude [dB]", "red")
fourier(noiseHandel, newSampleRate, "Fourier Transform of Noise + Handel signal",
        "Frequency [Hz]", "Amplitude [dB]", "red")
# Using the same carrier signal and k = 1
modulatedNoiseSignal = k1 * noiseHandel * carrierSignal
# graph modulated noise signal
graph(intervalResampledSignal, modulatedNoiseSignal, "Modulated noise signal", "Time [s]", "Amplitude [dB]", "coral")
fourier(modulatedNoiseSignal, newSampleRate, "Fourier Transform of Modulated Noise Signal",
        "Frequency [Hz]", "Amplitude [dB]", "coral")
# Demodulate the signal
demodulateNoiseSignal = modulatedNoiseSignal * carrierSignal
# Graph the Fourier transform of modulated signal * carrierSignal
fourier(demodulateNoiseSignal, newSampleRate, "Fourier Transform of AM Noise Signal x carrierSignal",
        "Frequency [Hz]", "Amplitude [dB]", "coral")
# Apply lowpass filter to recover the original signal
cutoffFrequency = 4096
cutoff = cutoffFrequency / nyquist  # nyquist = 0.5 * newSampleRate
# Create a butterworth filter
values = butter(8, cutoff, btype='lowpass')
b2, a2 = values[0], values[1]
# Apply the filter to the AM signals
filteredNoiseSignal = filtfilt(b2, a2, demodulateNoiseSignal)
# Graph the demodulated signal
graph(intervalResampledSignal, filteredNoiseSignal, "Demodulated AM noise signal", "Time [s]", "Amplitude [dB]",
      "coral")
# Writes the demodulated noise signal into a wav file
write("DemodulatedNoise.wav", newSampleRate, filteredNoiseSignal.astype(np.int16))
