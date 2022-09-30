from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import butter, resample, filtfilt
import matplotlib.pyplot as plt
import numpy as np

# Objetivo: Construir un gráfico usando como dato las entradas
# Entrada: Arreglo de valores de eje X, Arreglo de valores de eje y, string de titulo, string para eje x,
#          string para eje y, string para el color del gráfico
def graficar(x, y, titulo, xlabel, ylabel, color):
    plt.figure(titulo)
    plt.plot(x, y, color)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Objetivo: Construir un espectrograma usando como datos las entradas
# Entrada: Arreglo de valores de eje X, Arreglo de valores de eje y, string de titulo, string para eje x,
#          string para eje y.
def graficarEspectrograma(x, y, titulo, xlabel, ylabel):
    plt.figure(titulo)
    plt.specgram(x, Fs=y)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Lectura de las señales de audio generadas
frecuenciaA, signalA = read("ChristianMendez.wav")
frecuenciaB, signalB = read("IsraelArias.wav")

'''
Pregunta 2: Lea las señales de audio generadas y determine a qué corresponde cada uno de los parámetros retornados.
Al leer las señales de audio generadas con la función read se devuelven dos parametros
El primero es la frecuencia de muestreo (sampling rate) de la señal y
el segundo parámetro es un arreglo que contiene valores enteros que representan a la señal
'''
# Pregunta 3. Grafique las señales de audio en el tiempo.

# Señal 1
amplitudA = len(signalA)  # se obtiene el largo de la lista de amplitudes
duracionA = amplitudA / frecuenciaA  # Calcula la duración del audio grabado
tiempoA = np.linspace(0, duracionA, amplitudA)  # Se consigue el eje del tiempo

# Señal 2
amplitudB = len(signalB)  # se obtiene el largo de la lista de amplitudes
duracionB = amplitudB / frecuenciaB  # Calcula la duración del audio grabado
tiempoB = np.linspace(0, duracionB, amplitudB)  # Se consigue el eje del tiempo

# Se grafican los audios grabados

# Señal 1
graficar(tiempoA, signalA, "Audio N°1: Christian Méndez", "Tiempo [s]", "Amplitud [dB]", "red")

# Señal 2
graficar(tiempoB, signalB, "Audio N°2: Israel Arias", "Tiempo [s]", "Amplitud [dB]", "blue")

# Cálculo de la transformada de Fourier:
fourierA = fft(signalA)
fourierB = fft(signalB)

# Se calcula también el eje de las frecuencias para cada una de las transformadas
frecuenciasFourierA = fftfreq(amplitudA)
frecuenciasFourierB = fftfreq(amplitudB)

# Se grafican las señales luego de haberse aplicado la transformada de fourier
graficar(frecuenciasFourierA, abs(fourierA), "Transformada de fourier para Audio N°1", "Frecuencia [Hz]", "Amplitud [dB]", "red")
graficar(frecuenciasFourierB, abs(fourierB), "Transformada de fourier para Audio N°2", "Frecuencia [Hz]", "Amplitud [dB]", "blue")

# b. Al resultado del punto 4, calcule la transformada de Fourier inversa.
fourierInversaA = ifft(fourierA).real
fourierInversaB = ifft(fourierB).real

# c. Compare con la señal leída en el punto 1.
graficar(tiempoA, fourierInversaA, "Transformada de fourier inversa para resultado de audio N°1", "Tiempo [s]", "Amplitud [dB]", "red")
graficar(tiempoB, fourierInversaB, "Transformada de fourier inversa para resultado de audio N°2", "Tiempo [s]", "Amplitud [dB]", "blue")

# 5. Calcule y grafique el espectrograma de cada una de las señales. El espectrograma
# permite visualizar información en el dominio de la frecuencia y del tiempo a la vez.

graficarEspectrograma(signalA, frecuenciaA, "Espectrograma de audio N°1", "Tiempo [s]", "Frecuencia [Hz]")
graficarEspectrograma(signalB, frecuenciaB, "Espectrograma de audio N°2", "Tiempo [s]", "Frecuencia [Hz]")

# Se carga el ruido marrón
frecuenciaRuido, audioRuidoMarron = read("Ruido Marrón.wav")

# Se necesita tener ambas señales con la misma frecuencia de muestreo para poder trabajarlas juntas posteriormente.
# Lo que requiere aplicar un resampling
numeroDeMuestras = round(len(audioRuidoMarron) * float(frecuenciaB) / frecuenciaRuido)
ruidoMarronResampleado = resample(audioRuidoMarron, numeroDeMuestras)
# Ahora la frecuencia de sampleo del ruido marrón es igual a la de la señal B
frecuenciaRuido = frecuenciaB
# Se calcula la amplitud del ruido marron resampleado
amplitudRuidoMarron = len(ruidoMarronResampleado)
# Se calcula la duración del ruido marron resampleado
duracionRuidoMarron = amplitudRuidoMarron / frecuenciaRuido
# Se calcula el eje del tiempo
tiempoRuidoMarron = np.linspace(0, duracionRuidoMarron, amplitudRuidoMarron)

############################### Creación de señal ruidosa (Señal B + Ruido marron #####################################

# Suma del audio B con el ruido marron
ruidoMasAudioB = np.zeros(amplitudB) # Contendrá la suma entre el ruido y la señal de audio
for i in range(0, amplitudB):
    ruidoMasAudioB[i] = ruidoMarronResampleado[i] + signalB[i]

amplitudRuidoAudioB = len(ruidoMasAudioB)
duracionRuidoAudioB = amplitudRuidoAudioB / frecuenciaB
tiempoRuidoAudioB = np.linspace(0, duracionRuidoAudioB, amplitudRuidoAudioB)

# Se escribe el audio de salida
write("IsraelAriasRuidoMarron.wav", frecuenciaB, ruidoMasAudioB.astype(np.int16))

# Se grafica el ruido marrón
graficar(tiempoRuidoMarron, ruidoMarronResampleado, "Gráfico de audio de la señal ruido marrón", "Tiempo [s]", "Amplitud [dB]", "brown")

# Se calcula la transformada de fourier del ruido marron
fourierRuidoMarron = fft(ruidoMarronResampleado)

# Se calcula también el eje de las frecuencias para la transformada
frecuenciasRuidoMarron = fftfreq(amplitudRuidoMarron)

# Se grafica la señal luego de haberle aplicado la transformada de fourier
graficar(frecuenciasRuidoMarron, abs(fourierRuidoMarron), "Transformada de fourier ruido marrón", "Frecuencia [Hz]", "Amplitud [dB]", "brown")

# Se calcula la transformada de fourier inversa para el ruido marron
fourierInversaRuidoMarron = ifft(fourierRuidoMarron).real

# Se grafica la transformada inversa para el ruido marron
graficar(tiempoRuidoMarron, fourierInversaRuidoMarron, "Transformada de fourier inversa para ruido marrón", "Tiempo [s]", "Amplitud [dB]", "brown")

# Se grafica el espectrograma para el ruido marron
graficarEspectrograma(ruidoMarronResampleado, frecuenciaRuido, "Espectrograma de ruido marrón", "Tiempo [s]", "Frecuencia [Hz]")

# Se grafica la señal ruidosa (audio B + ruido marron)
graficar(tiempoRuidoAudioB, ruidoMasAudioB, "Gráfico de audio de la señal B + ruido marron", "Tiempo [s]", "Amplitud [dB]", "orange")

# Se calcula la transformada de fourier de la señal ruidosa (audio B + ruido marron)
fourierRuidoAudioB = fft(ruidoMasAudioB)

# Se calcula también el eje de las frecuencias para la transformada
frecuenciasRuidoAudioB = fftfreq(amplitudRuidoAudioB)

# Se grafica la señal ruidosa (audio B + ruido marrón) luego de haberle aplicado la transformada de fourier
graficar(frecuenciasRuidoAudioB, abs(fourierRuidoAudioB), "Transformada de la señal B + ruido marron", "Frecuencia [Hz]", "Amplitud [dB]", "orange")

# Se calcula la transformada de fourier inversa para el audio más ruido marrron
fourierInversaRuidoAudioB = ifft(fourierRuidoAudioB).real

# Se grafica la transformada inversa para la señal ruidosa (audio B + ruido marron)
graficar(tiempoRuidoAudioB, fourierInversaRuidoAudioB, "Transformada de fourier inversa de la señal B + ruido marron", "Tiempo [s]", "Amplitud [dB]", "orange")

# Se grafica el espectrograma para la señal B + ruido marron
graficarEspectrograma(ruidoMasAudioB, frecuenciaB, "Espectrograma de la señal B + ruido marrron", "Tiempo [s]", "Frecuencia [Hz]")

# Se prueban 3 filtros IIR con valores distintos

nyquist = 0.5 * frecuenciaB  # Frecuencia de Nyquist, usada para calcular los parametros de entrada al filtro

corte1 = 500
corte_normal1 = corte1 / nyquist
corte1_2 = 4975
corte_normal1_2 = corte1_2 / nyquist
b, a = butter(8, [corte_normal1,corte_normal1_2], btype='bandpass')

corte2 = 5000
corte_normal2 = corte2 / nyquist
b2, a2 = butter(8, corte_normal2, btype='lowpass')

corte3 = 8000
corte_normal3 = corte3 / nyquist
b3, a3 = butter(8, corte_normal3, btype='highpass')

audioFiltrado1 = filtfilt(b, a, ruidoMasAudioB)
audioFiltrado2 = filtfilt(b2, a2, ruidoMasAudioB)
audioFiltrado3 = filtfilt(b3, a3, ruidoMasAudioB)

write("AudioFiltrado1.wav", frecuenciaB, audioFiltrado1.astype(np.int16))
write("AudioFiltrado2.wav", frecuenciaB, audioFiltrado2.astype(np.int16))
write("AudioFiltrado3.wav", frecuenciaB, audioFiltrado3.astype(np.int16))

# Se toma el audio filtrado 1 de bandpass para calcular y graficar su transformada de fourier y espectrogrrama
amplitudAudioFiltrado1 = len(audioFiltrado1)

# Cálculo de la transformada de Fourier:
fourierAudioFiltrado1 = fft(audioFiltrado1)
# Se calcula la duración del audio filtrado
duracionAudioFiltrado1 = amplitudAudioFiltrado1 / frecuenciaB  # Calcula la duración del audio grabado
# Se consigue el eje del tiempo
tiempoAudioFiltrado1 = np.linspace(0, duracionAudioFiltrado1, amplitudAudioFiltrado1)  # Se consigue el eje del tiempo

# Se calcula también el eje de las frecuencias
frecuenciasFourierAudioFiltrado1 = fftfreq(amplitudAudioFiltrado1)

# Se grafica la señal luego de haberle aplicado la transformada de fourier
graficar(frecuenciasFourierAudioFiltrado1, abs(fourierAudioFiltrado1), "Transformada de fourier para Audio Filtrado", "Frecuencia [Hz]", "Amplitud [dB]", "purple")
# Se grafica el espectrograma
graficarEspectrograma(audioFiltrado1, frecuenciaB, "Espectrograma de Audio filtrado", "Tiempo [s]", "Frecuencia [Hz]")