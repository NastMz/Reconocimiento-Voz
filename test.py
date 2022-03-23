import os
import wave
import pyaudio
from scipy.io import wavfile

import calculate as calc
import command as cmd

nparts = 100  # numero de partes en las que se dividieron las grabaciones en el entrenamiento

# DEFINIMOS PARÁMETROS
FORMAT = pyaudio.paInt16  # el formato de los samples
CHANNELS = 1  # número de canales
RATE = 44200  # 44200 frames por segundo
CHUNK = 1024  # unidades de memoria menores que se almacenará durante la transmisión de datos
duration = 2  # duración en segundos de nuestra grabación


# -------------------------------- IGNORAR ---------------------------------------
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

while True:
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # INICIAMOS GRABACIÓN
    frames = []

    # input("Presiona enter para hablar...")
    print("escuchando...")
    for k in range(0, int(RATE / CHUNK * duration)):
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

    fs, data = wavfile.read("grabacion.wav")
    if max(data) >= 8000:
        ffts = calc.calculate_fft_record("grabacion.wav")  # se calcula la transformada de fourier de la grabación

        parts = calc.split(ffts, nparts)  # se divide la grabacion

        energy_sequence = []
        for k in range(0, nparts):
            energy_sequence.append(
                calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de cada parte del vector

        command = cmd.find_command(energy_sequence)  # se busca el posible comando en base a la energia calculada

        os.remove("grabacion.wav")  # se elimina el archivo creado para volver a iniciar el proceso

        print(command)