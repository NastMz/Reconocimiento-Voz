import numpy as np
import calculate as calc

n = 150  # número de grabaciones por palabra
nparts = 100  # número de partes en las que se divide el vector
folder = "train_data/"  # carpeta en la que se guardan las grabaciones
commands = ["arriba", "abajo", "derecha", "izquierda"]


# FUNCIÓN PARA CALCULAR EL PROMEDIO DE ENERGÍA DE CADA COMANDO
def train(records):
    ffts = []  # vector para almacenar las transformadas de fourier de las grabaciones
    energies = []  # vector para almacenar las energias de las partes
    energy_sequence = []  # vector para almacenar las energias promes de los cmandos

    # CICLO PARA REALIZAR EL PROCESO CON CADA UNA DE LAS GRABACIONES
    for j in range(0, len(records)):
        fft = calc.calculate_fft_record(records[j])  # se calcula la transformada de fourier de la grabación
        ffts.append(fft)  # se guarda la transformada en el vector

        # se divide el vector de la transformada de fourier de la grabación en n partes
        parts = calc.split(ffts[j], nparts)

        # CALCULAR LAS ENERGÍAS DE LAS PARTES DE LA TRANSFORMADA DE FOURIER
        # Se calcula la transformada de cada parte una de las partes para cada grabacion
        energy_part = []
        for k in range(0, nparts):
            energy_part.append(calc.calculate_energy(parts[k]))
        energies.append(energy_part)

    # CALCULAR LOS PROMEDIOS DE LAS ENERGÍAS
    # Como en el vector energies estan las energias de las partes de cada grabacion
    # se saca el promedio de todas
    for m in range(0, len(energies[0])):
        e = []
        for o in range(0, len(energies)):
            e.append(energies[o][m])
        energy_sequence.append(np.mean(e))

    return energy_sequence


# VECTOR DE GRABACIONES
records_up = []
records_down = []
records_left = []
records_right = []

# Se buscan todas las grabaciones que existen y se guarda el listado en un vector
for i in range(1, n + 1):
    records_up.append(folder + commands[0] + str(i) + ".wav")
    records_down.append(folder + commands[1] + str(i) + ".wav")
    records_right.append(folder + commands[2] + str(i) + ".wav")
    records_left.append(folder + commands[3] + str(i) + ".wav")

# Con el listado de grabaciones se calcula la energia de cada comando
energy_up = train(records_up)
energy_down = train(records_down)
energy_left = train(records_left)
energy_right = train(records_right)

print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra ARRIBA es: ")
print(energy_up)
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra ABAJO es: ")
print(energy_down)
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra DERECHA es: ")
print(energy_right)
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra IZQUIERDA es: ")
print(energy_left)
print("-----------------------------------------------------------------")
