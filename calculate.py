import numpy as np
import scipy.fft
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq


# FUNCIÓN PARA CALCULAR LA TRANSFORMADA RÁPIDA DE FOURIER DE UNA GRABACIÓN
def calculate_fft_record(audio):
    fs, data = wavfile.read(audio)
    index1 = 0
    while 2000 > data[index1] > -2000:
        index1 += 1
    part = data[index1:]
    index2 = len(part) - 1
    while 2000 > part[index2] > -2000:
        index2 -= 1
    split_record = part[:index2]
    fft = scipy.fft.rfft(split_record)
    return fft


# FUNCIÓN PARA DIVIDIR EL VECTOR DE LA TRANSFORMADA RÁPIDA DE FOURIER EN N PARTES IGUALES
def split(array, parts):
    while len(array) % parts != 0:
        array = np.append(array, 0)
    return np.split(array, parts)


# FUNCIÓN PARA CALCULAR LA ENERGÍA DE LAS PARTES DEL VECTOR DE LA TRANSFORMADA RÁPIDA DE FOURIER
def calculate_energy(array):
    suma = 0
    n = len(array)
    for i in range(0, n):
        suma = pow(np.abs(array[i]), 2) + suma
    energy = (1 / n) * suma
    return energy
