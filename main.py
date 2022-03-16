import os
import struct
import wave

import numpy as np
import pyaudio

import calculate as calc
import command as cmd

nparts = 600

# DEFINIMOS PARÁMETROS
FORMAT = pyaudio.paInt16  # el formato de los samples
CHANNELS = 1  # número de canales
RATE = 44200  # 44200 frames por segundo
CHUNK = 1024  # unidades de memoria menores que se almacenará durante la transmisión de datos

# audio = pyaudio.PyAudio()
#
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

while True:
    # print("Empieza a hablar")
    #
    # data = stream.read(CHUNK)
    #
    # frame = np.frombuffer(np.array(data), dtype=np.int16)
    #
    # fft = calc.calculate_fft(frame)
    # parts = calc.split(fft, nparts)
    #
    # energy_sequence = []
    # for k in range(0, nparts):
    #     energy_sequence.append(calc.calculate_energy(parts[k]))
    #
    # print(cmd.find_command(energy_sequence))

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # INICIAMOS GRABACIÓN
    frames = []

    print("escuchando...")
    for k in range(0, int(RATE / CHUNK * 2)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("fin escucha")
    # DETENEMOS GRABACIÓN
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # CREAMOS/GUARDAMOS EL ARCHIVO DE AUDIO

    waveFile = wave.open("grabacion.wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    ffts = calc.calculate_fft_record("grabacion.wav")  # se calcula la transformada de fourier de la grabación

    parts = calc.split(ffts, nparts)

    energy_sequence = []
    for k in range(0, nparts):
        energy_sequence.append(
            calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de la parte uno del vector
        # de la transformada de fourier de la grabación

    command = cmd.find_command(energy_sequence)
    print(command)

    os.remove("grabacion.wav")
