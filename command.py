import numpy as np
import calculate as calc

folder = "records/"
commands = ["arriba", "abajo", "derecha", "izquierda"]
n = 100  # número de grabaciones por palabra
nparts = 10  # numero de partes en las que se dividieron las grabaciones en el entrenamiento

a = b = c = d = 0

# Vectores de secuencia de energia promedio de cada comando
arriba = np.array()

abajo = np.array()

derecha = np.array()

izquierda = np.array()


# FUNCIÓN PARA ENCONTRAR EL COMANDO EN BASE A LA ENERGIA
def find_command(energy_seq):
    # Se restan la energia promedio de cada comando con la de la que se capta en tiempo real
    # Para ver cual tiene menor diferencia
    differences = []
    up = abs(arriba - energy_seq)
    down = abs(abajo - energy_seq)
    right = abs(derecha - energy_seq)
    left = abs(izquierda - energy_seq)

    # se separan las diferencias por parte
    # Por ejemplo si tuvieramos estas diferencias:
    # up = [1,2,3,4]
    # down = [5,6,7,8]
    # right = [9,1,2,3]
    # left = [4,5,6,7]
    # Entonces este for haria lo siguiente:
    # differences = [[1,5,9,4],[2,6,1,6],[3,7,2,7],[4,8,3,7]]
    # Es decir que crea un vector que tiene dentro vectores con las diferencias de cada parte
    # independientemente de el comando
    for i3 in range(0, nparts):
        difference = [up[i3], down[i3], right[i3], left[i3]]
        differences.append(difference)

    # luego con el vector que se creo se busca cual fue el menor de cada parte
    # siguiendo con el ejemplo anterior:
    # differences = [[1,5,9,4],[2,6,1,6],[3,7,2,7],[4,8,3,7]]
    # se busca el menor de cada uno y se guarda la posicion,
    # para la primera parte el menor es 1 por eso se guardaria 0 que es la posicione en la que esta,
    # entonces quedaria asi:
    # min_differences = [0,2,2,2]
    min_differences = []
    for i4 in range(0, 4):
        index = differences[i4].index(min(differences[i4]))
        min_differences.append(index)

    # Ahora se maneja una especie de sistema de puntos para encontar el que tuvo menor diferencia
    # Para esto entonces se cuenta cuantas veces aparece una posicion
    # En el caso del ejemplo, ya que tenemos 4 posiciones entonces contamos cuantas veces estan
    # 0, 1, 2, 3 en el vector min_differences:
    # count = [1, 0, 3, 0]
    count = []
    for i5 in range(0, 4):
        count.append(min_differences.count(i5))

    # Luego se mira quien fue el que tuvo mas "puntos" y se toma ese como el comando que probablemente
    # se dijo
    return commands[count.index(max(count))]

# -------------------------------- IGNORAR ---------------------------------------
# Esto sirve para comprobar que tan exacto es el algoritmo despues de entrenarlo,
# usando las grabaciones como si fuera alguien hablando en tiempo real
#
# for i in range(0, len(commands)):
#     print("_______________" + commands[i].upper() + "_______________")
#     for j in range(1, n + 1):
#         file = folder + commands[i] + str(j) + ".wav"
#         ffts = calc.calculate_fft_record(file)  # se calcula la transformada de fourier de la grabación
#
#         parts = calc.split(ffts, nparts)
#
#         energy_sequence = []
#         for k in range(0, nparts):
#             energy_sequence.append(calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de la parte uno
#             # del vector
#             # de la transformada de fourier de la grabación
#
#         command = find_command(energy_sequence)
#
#         if command == "arriba":
#             a = a + 1
#         elif command == "abajo":
#             b = b + 1
#         elif command == "derecha":
#             c = c + 1
#         elif command == "izquierda":
#             d = d + 1
#
#     print("arriba: " + str(a))
#     print("abajo: " + str(b))
#     print("derecha: " + str(c))
#     print("izquierda: " + str(d))
#     a = b = c = d = 0
