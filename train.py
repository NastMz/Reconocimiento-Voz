import numpy as np
import calculate as calc

n = 110  # número de grabaciones por palabra
nparts = 600
folder = "records/"
commands = ["arriba", "abajo", "derecha", "izquierda"]


def train(records):
    ffts = []  # vector para almacenar las transformadas de fourier de las grabaciones
    energies = []
    energy_sequence = []
    # CICLO PARA REALIZAR EL PROCESO CON CADA UNA DE LAS GRABACIONES
    for j in range(0, len(records)):
        fft = calc.calculate_fft_record(records[j])
        ffts.append(fft)  # se calcula la transformada de fourier de la grabación

        # se divide el vector de la transformada de fourier de la grabación en n partes
        parts = calc.split(ffts[j], nparts)

        energy_part = []
        for k in range(0, nparts):
            energy_part.append(calc.calculate_energy(parts[k]))
        energies.append(energy_part)

        # CALCULAR LOS PROMEDIOS DE LAS ENERGÍAS DE LAS TRANSFORMADAS DE FOURIER DE LAS GRABACIONES
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

for i in range(1, n + 1):
    records_up.append(folder + commands[0] + str(i) + ".wav")
    records_down.append(folder + commands[1] + str(i) + ".wav")
    records_right.append(folder + commands[2] + str(i) + ".wav")
    records_left.append(folder + commands[3] + str(i) + ".wav")

print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra ARRIBA es: ")
print(train(records_up))
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra ABAJO es: ")
print(train(records_down))
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra DERECHA es: ")
print(train(records_right))
print("-----------------------------------------------------------------")
print("La secuencia de umbrales de energía para la palabra IZQUIERDA es: ")
print(train(records_left))
print("-----------------------------------------------------------------")