# Imports
import matplotlib.pyplot as plt
import numpy as np

"""
Funciones
"""


# Entrada: arreglo que corresponde al eje X,
#          arreglo que corresponde al eje Y,
#          String del nombre eje x, String del nombre eje Y, 
#          String del titulo del gráfico 
# Salida: vacio
# Objetivo: Realizar un gráfico
def graficar(eje_x, eje_y, xlabel, ylabel, titulo):
    plt.plot(eje_x, eje_y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.show()


# Entrada: arreglo que corresponde a la señal modulada OOK
#          entero que corresponde al valor de la SNR,
# Salida: arreglo que contiene la señal modulada con ruido awgn agregado
# Objetivo: Simular la transmisión de una señal modulada OOK por un canal AWGN
def noise(lista, snr_db):
    ruido = np.random.normal(0, 1, len(lista))
    p_signal = np.sum(np.abs(lista) * np.abs(lista))
    p_noise = np.sum(np.abs(ruido) * np.abs(ruido))
    snr_lineal = np.exp(snr_db / 10)
    sigma = np.sqrt(p_signal / (p_noise * snr_lineal))
    ruido = sigma * ruido
    awgn = lista + ruido
    return awgn


# Entrada: Cadena de bits (Arreglo con 1 o 0), tasa de bits,
#          frecuencia del coseno señal bit 1 (entero), 
#          cantidad de muestras por tiempo de bit (entero). 
# Salida: Señal modulada y tiempo (arreglos)
# Obj: Modular una cadena de bits en OOK
def modular(cadena_bits, tasa_bits, frecuencia, frec_sampleo):
    tiempo_bit = 1 / tasa_bits
    tiempo = np.linspace(0, tiempo_bit, frec_sampleo)
    # Arreglos para la señal 0 y señal 1
    bit_zero_signal = tiempo * 0
    bit_one_signal = np.cos(2 * np.pi * frecuencia * tiempo)
    # Arreglos de salida
    tiempo_salida = []
    signal_salida = []
    # Lectura de cada uno de los bits de la cadena
    for k, bit in enumerate(cadena_bits):
        tiempo_salida.extend(tiempo + k * tiempo_bit)
        if bit == 0:  # En caso de leer un 0 almacena la señal de 0
            signal_salida.extend(bit_zero_signal)
        else:  # Si no almacena la señal del bit 1
            signal_salida.extend(bit_one_signal)
    tiempo_salida = np.array(tiempo_salida)
    return signal_salida, tiempo_salida


# Entrada: Arreglo de la señal modulada, frecuencia del coseno
#          utilizado para la señal que representa al bit 1 (entero),
#          Arreglo de tiempo de la señal modulada, tasa de bit (entero)
# Salida: Arreglo de la cadena de bits correspondiente a la 
#         señal demodulada
# Obj: demodular una señal modulada
def demodular(signal, frecuencia, tiempo, bit_rate):
    signal_temp = signal * np.cos(2 * np.pi * frecuencia * tiempo)
    tiempo_bit = 1 / bit_rate
    numero_bits = int(tiempo[-1] / tiempo_bit)
    posiciones_por_bit = int(len(tiempo) / numero_bits)
    ini = 0
    fin = posiciones_por_bit
    bits_salida = []
    for k in range(0, numero_bits):
        acum = np.sum(signal_temp[ini:fin])
        ini += posiciones_por_bit
        fin += posiciones_por_bit
        if acum < 10:
            bits_salida.append(0)
        else:
            bits_salida.append(1)
    return bits_salida


# Entrada: arreglo de bits que fue transmitido originalmente,
#          arreglo de bits que fue demodulado.
# Salida: numero correspondiente a la tasa de errores de bits
# Obj: Calcula el BER dado el arreglo de bits original y el arreglo de bits demodulado
def calcular_ber(arreglo_original, arreglo_demodulado):
    cantidad_bits_erroneos = 0
    for k in range(0, len(arreglo_original)):
        if arreglo_original[k] != arreglo_demodulado[k]:
            cantidad_bits_erroneos = cantidad_bits_erroneos + 1
    # BER = Cantidad de bits erroneos / Cantidad de bits totales
    tasa_errores = cantidad_bits_erroneos / len(arreglo_original)
    return tasa_errores


"""
Bloque principal
"""

# Entradas
tasaBits = 100  # Bit rate

# Párametros de la modulación
tiempoBit = 1 / tasaBits
fs = 30  # Frecuencia de sampleo
fc = tasaBits * 2  # Frecuencia del coseno

"""
Ejemplo de modulación/demodulación con un ejemplo pequeño
"""
cadenaBits = [1, 0, 0, 1]  # Arreglo de bits
print("Cadena de bits de entrada: ", cadenaBits)
# Modulación OOK
salida, tiempoSalida = modular(cadenaBits, tasaBits, fc, fs)
graficar(tiempoSalida, salida, 'Tiempo[s]', 'Amplitud', 'Señal modulada OOK')
# Demodulación 
demodulada = demodular(salida, fc, tiempoSalida, tasaBits)
print("Cadena de bits demodulada: ", demodulada)

"""
Simulación sistema de comunicación
"""
# Cadena de bits aleatoria
arreglo_aleatorio = np.random.randint(2, size=100000)

plt.figure('Bit Error Rate v/s SNR')
colores = ['red', 'green', 'blue']
for i in range(0, 3):
    # Variacion en la tasa de bits
    tasaBits = tasaBits + 1000 * i
    print("Simulando para tasa de bit: ", tasaBits)
    fc = 2 * tasaBits
    berArray = []
    snrArray = []
    # Se modula el arreglo de bits con OOK/ASK
    salida, tiempoSalida = modular(arreglo_aleatorio, tasaBits, fc, fs)
    # se simula la variacion de snr
    for snr in range(-2, 11):
        snrArray.append(snr)
        # Se agrega el ruido provocado por el canal AWGN a la señal modulada
        awgnSignal = noise(salida, snr)
        # Se demodula la señal con ruido awgn
        demoduladaAwgn = demodular(awgnSignal, fc, tiempoSalida, tasaBits)
        ber = calcular_ber(arreglo_aleatorio, demoduladaAwgn)
        berArray.append(ber)
    # Generar gráfico de resumen
    plt.plot(snrArray, berArray, colores[i], marker='o', label=str(tasaBits) + ' [bps]')

plt.xlabel('SNR [dB]')
plt.ylabel('BER')
plt.yscale('log')
plt.xscale('linear')
plt.title('BER vs SNR')
plt.legend()
plt.grid(True)
plt.show()
