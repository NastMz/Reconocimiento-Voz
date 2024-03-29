import os
import wave
import pyaudio
from scipy.io import wavfile

import calculate as calc
import command as cmd

num_parts = 100  # numero de partes en las que se dividieron las grabaciones en el entrenamiento

# DEFINIMOS PARÁMETROS
FORMAT = pyaudio.paInt16  # el formato de los samples
CHANNELS = 1  # número de canales
RATE = 44200  # 44200 frames por segundo
CHUNK = 1024  # unidades de memoria menores que se almacenará durante la transmisión de datos
duration = 2  # duración en segundos de nuestra grabación


# -------------------------------- IGNORAR ---------------------------------------
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

# while True:

def voice_recognition():
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

    wave_file = wave.open("grabacion.wav", 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    fs, data = wavfile.read("grabacion.wav")
    if max(data) >= 10000:
        ffts = calc.calculate_fft_record("grabacion.wav")  # se calcula la transformada de fourier de la grabación

        parts = calc.split(ffts, num_parts)  # se divide la grabacion

        energy_sequence = []
        for k in range(0, num_parts):
            energy_sequence.append(
                calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de cada parte del vector

        command = cmd.find_command(energy_sequence)  # se busca el posible comando en base a la energia calculada

        os.remove("grabacion.wav")  # se elimina el archivo creado para volver a iniciar el proceso

        return command
