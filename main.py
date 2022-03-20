import os
import wave
import pyaudio

import calculate as calc
import command as cmd

nparts = 100  # numero de partes en las que se dividieron las grabaciones en el entrenamiento
e_range = [10800000000000,
           48000000000000]  # rango de energias para buscar un comando, si no esta en ese rango se ignora la grabacion
# DEFINIMOS PARÁMETROS
FORMAT = pyaudio.paInt16  # el formato de los samples
CHANNELS = 1  # número de canales
RATE = 44200  # 44200 frames por segundo
CHUNK = 1024  # unidades de memoria menores que se almacenará durante la transmisión de datos

# -------------------------------- IGNORAR ---------------------------------------
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK)

while True:

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # INICIAMOS GRABACIÓN
    frames = []

    input("Presiona enter para hablar...")
    print("escuchando...")
    for k in range(0, int(RATE / CHUNK)):
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

    parts = calc.split(ffts, nparts)  # se divide la grabacion

    energy_sequence = []
    for k in range(0, nparts):
        energy_sequence.append(
            calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de cada parte del vector

    if energy_sequence[0] > e_range[0] and energy_sequence[nparts - 10] < e_range[1]:
        command = cmd.find_command(energy_sequence)  # se busca el posible comando en base a la energia calculada
        print("El comando probablemente es: " + command)

    os.remove("grabacion.wav")  # se elimina el archivo creado para volver a iniciar el proceso

    # -------------------------------- IGNORAR ---------------------------------------
    # prueba de captar la voz en tiempo real para no usar grabaciones
    #
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
