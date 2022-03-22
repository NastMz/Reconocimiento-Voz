# IMPORTAMOS LAS LIBRERíAS NECESARIAS
import pyaudio
import wave

commands = ["arriba", "abajo", "derecha", "izquierda"]
n = 25  # número de grabaciones por palabra

# DEFINIMOS PARÁMETROS
FORMAT = pyaudio.paInt16  # el formato de los samples
CHANNELS = 1  # número de canales
RATE = 44200  # 44200 frames por segundo
CHUNK = 1024  # unidades de memoria menores que se almacenará durante la transmisión de datos
duration = 2  # duración en segundos de nuestra grabación

for i in range(0, 4):
    print("Pronuncia la palabra " + commands[i].upper() + " cuando veas el mensaje \"grabando'" + commands[
        i].upper() + "'...\"")
    input("Presiona enter para empezar...")
    for j in range(1, n + 1):
        file = "records/" + commands[i] + str(j) + ".wav"  # nombre que tendrá el archivo de sonido
        # INICIAMOS "pyaudio"
        audio = pyaudio.PyAudio()

        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        # INICIAMOS GRABACIÓN
        print("grabando '" + commands[i].upper() + "'...")
        frames = []

        for k in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("grabación terminada")

        # DETENEMOS GRABACIÓN
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # CREAMOS/GUARDAMOS EL ARCHIVO DE AUDIO

        waveFile = wave.open(file, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        print("-----------------------------")
        input("Presiona enter para continuar...")

    print("GRABACIONES DE " + commands[i].upper() + " FINALIZADAS")
    input("Presiona enter para continuar...")

print("GRABACIONES FINALIZADAS")
