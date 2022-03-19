import numpy as np
from scipy.io import wavfile


# FUNCIÓN PARA CALCULAR LA TRANSFORMADA RÁPIDA DE FOURIER DE UN FRAME
def calculate_fft(data):
    fft = np.fft.fft(data)
    return fft


# FUNCIÓN PARA CALCULAR LA TRANSFORMADA RÁPIDA DE FOURIER DE UNA GRABACIÓN
def calculate_fft_record(audio):
    fs, data = wavfile.read(audio)
    fft = np.fft.fft(data)
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
        suma = pow(abs(array[i]), 2) + suma
    energy = (1 / n) * suma
    return energy
