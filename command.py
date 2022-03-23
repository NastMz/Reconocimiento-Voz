import numpy as np
import calculate as calc

folder = "train_data/"
commands = ["UP", "DOWN", "RIGHT", "LEFT"]
n = 150  # número de grabaciones por palabra
nparts = 100  # numero de partes en las que se dividieron las grabaciones en el entrenamiento

# Vectores de secuencia de energia promedio de cada comando
arriba = np.array(
    [17037238729121.74, 12726484125549.36, 2870896908921.4434, 1110226111112.9788, 929998047322.9152, 295026036059.2352,
     74926761317.69656, 137128315158.6379, 16081126020.370876, 3879820389.332464, 5126844558.6951065,
     3777095478.8673153, 2858226382.49412, 3032314929.881241, 1678320715.9913127, 2187031456.72968, 3101382457.472757,
     757001286.0865669, 498391680.3746083, 439813580.0399152, 259597682.4603491, 172057777.60755217, 113884438.43800402,
     80960631.97618362, 73631241.5151404, 76390699.05348235, 84559528.1595232, 95134684.89689215, 94734976.65756114,
     104115127.71932255, 120427398.5919961, 149582576.58565816, 193585396.63691175, 229250394.9467897,
     227511776.03216162, 253145737.07483572, 229159068.71414968, 182716996.53200775, 158740825.42966062,
     125630556.25920753, 103513820.6198518, 103601989.32017048, 108646519.89958133, 124189389.60665184,
     137701864.40004617, 130495733.02698646, 116520643.60857578, 101321719.9231008, 76253230.40502152,
     62327458.21334902, 62812341.121187314, 77148342.77226686, 100216115.24102409, 120540278.72354212,
     129259168.61266449, 137075218.8092618, 122384523.81157577, 111827684.78313456, 100427723.53125621,
     106558573.99092847, 121817158.60011688, 164340020.82927078, 187185665.195349, 221541988.30097017,
     257068255.1788562, 232210784.25417545, 223090548.13557997, 190574673.0554356, 151931997.2849874,
     118628853.37380467, 103098065.45567825, 93949097.87636557, 96013440.15430939, 84211784.89007734, 75775770.31382349,
     73926836.76324067, 81544775.24923572, 114819849.83944377, 173959929.8080523, 277860356.6940531, 427534118.05244786,
     504243655.01701903, 788652260.3155961, 3183040989.113832, 2138405006.144511, 1686901244.5823035, 3032780696.144117,
     2909222114.324914, 3847671406.619606, 5054626492.378363, 3954193555.9790154, 17729106677.195396,
     139499713695.77792, 75803959625.8669, 310790096076.40717, 922774840914.6078, 1162367609488.5288,
     3217802245234.0776, 13148375086445.297, 16202524714722.934])

abajo = np.array(
    [6526911157981.352, 21529848345396.227, 7855891105351.97, 918264959837.033, 25456125102.24672, 94743659543.7372,
     97519799403.41624, 37557790898.81181, 7029606670.230202, 5390740561.210776, 5578871474.364592, 1684246014.3866181,
     2243291689.2592416, 1299328365.4043715, 682626120.647022, 784081777.3604962, 1109749585.3628812,
     428657144.82159585, 263001899.5748774, 250618320.6944488, 224965805.37002012, 213439264.42086193,
     203936622.81123468, 194866262.80879065, 184186874.76444116, 171614092.26565263, 159577098.44936392,
     145262524.66326585, 130565213.90207097, 117767996.79487008, 104915978.43949316, 95361788.03467897,
     89202456.87446044, 84786423.95534162, 83357576.94524303, 83118292.89156288, 83181391.37643787, 83529100.54819877,
     83076276.36425082, 80928350.33057421, 76919828.51729053, 70754990.91503969, 62409323.7138854, 52415027.90986637,
     41327177.54880296, 29889714.884163685, 19418463.93770555, 10633436.881500982, 4358294.790473752, 964278.4516246287,
     1060892.2771663389, 4593079.093234677, 11052680.210552935, 19881010.83096116, 30510757.665369052,
     41937234.26609253, 52957587.06197631, 63005004.8591714, 71155136.45282438, 77204861.55698624, 81146763.16152216,
     83323213.35830574, 83562938.16568743, 83244380.75882831, 83117762.0566945, 83171310.65101095, 84809316.05384097,
     89267606.69731115, 95684698.95311351, 105022643.10553104, 118052552.64905915, 131325996.18791807,
     145518301.25501573, 160014254.70447809, 171693625.73106793, 184230681.08913848, 195133325.69395515,
     204344294.15513423, 214453158.14454547, 225446051.59154254, 252085132.11392817, 263044957.80285498,
     451383157.1754309, 1121548956.56019, 767032354.8009993, 690413510.6176798, 1327485843.3635473, 2244241285.948979,
     1733866755.4543211, 5732487420.508464, 5225996005.260346, 7744758195.956468, 38180142619.65023, 99094727847.2916,
     92762581555.43768, 26196501281.680405, 1097904626217.5292, 8726626337000.767, 20635574727635.094,
     6369002226254.493])

derecha = np.array([20814970398471.227, 11314599313911.664, 305877042246.1106, 405038579012.80896, 792174039104.9379,
                    470424465928.0734, 140810669112.50424, 277067797614.2315, 267162585245.32724, 59341421312.17477,
                    50098875693.89202, 19316376042.564354, 19561631189.536568, 18532175390.19898, 8546454260.216949,
                    22948003019.376934, 34656658776.56513, 3212284600.3243246, 629295095.7186569, 620004599.9035646,
                    279661141.9532113, 294032287.99034315, 274628246.80373293, 233818729.24636424, 262911105.8130771,
                    190858904.18068916, 259936160.6149389, 161501205.29078972, 229311706.3716243, 146032211.73075277,
                    198128589.0240127, 143493476.44452104, 156117308.79842165, 141323197.5924675, 119809899.07442762,
                    138757408.16469225, 91210080.19439211, 129055709.4244001, 75152610.98591077, 111902151.99731684,
                    68343802.88020185, 92509837.97579771, 68217763.50698142, 74541802.09297723, 67758533.76101227,
                    63202090.701005846, 66746143.07533052, 58250379.607486434, 62406631.86494106, 58671929.694096945,
                    58771895.273655474, 62282438.812816165, 58779643.932297096, 66008596.068433955, 64449352.582521744,
                    66409755.15870056, 76377212.74397609, 66687282.435568646, 94255412.32097268, 67345181.05180626,
                    112702571.96804857, 75841488.78939329, 127928449.05329959, 93936486.20783195, 135820313.3033093,
                    124327802.97025165, 136862441.76180744, 161341874.8368452, 139656550.46713436, 202240960.05234692,
                    143805818.74041328, 230673834.51010168, 162816884.6608309, 257832635.1240688, 195096278.56901503,
                    258217362.05109084, 240475202.89806357, 268778350.1094002, 301199526.43286127, 278567779.02270544,
                    632423159.0488137, 636355057.8519956, 3736066067.925589, 35488425362.322105, 22007696728.095867,
                    8617714767.063187, 18815781351.294838, 19212675883.038025, 20548992207.375065, 52443185337.57671,
                    57709040734.86094, 288510887971.9565, 256789557717.1497, 147287458922.83334, 477721090602.7037,
                    823755652683.8817, 360282246277.709, 333084350348.98773, 12516446327453.387, 19581794253833.56])

izquierda = np.array(
    [28539239221376.758, 13821463842145.404, 514160062254.57196, 450967971182.9384, 355386500157.57, 501399450213.7802,
     251619778979.7048, 228454269119.00266, 118003396782.94838, 117638520796.39615, 248380754726.4763,
     331684127683.33124, 249371906093.58853, 217128869976.55197, 178862930904.00275, 351872590493.2671, 385630206558.65,
     124392338164.7127, 37017889691.41006, 4745139104.014445, 174132247.91374734, 18487312.434331305, 13591530.39823451,
     12202504.67766853, 9603667.95389382, 9102626.636636881, 8407469.569807457, 7434888.608832521, 6794251.023148993,
     5956493.9699715, 5540491.233941394, 5110143.135534026, 4836554.914907382, 4517643.818540332, 4529935.515359979,
     4343625.777095349, 4339486.253576308, 4061305.762637865, 3610528.1945888526, 3582072.9628695687,
     3489378.2417994426, 3565025.52111884, 3576770.010628012, 3520559.1162691535, 3452760.124424627, 3437669.0168226655,
     3395221.3646298805, 3414327.3434847123, 3422789.0441945232, 3466719.247069458, 3486479.6823955895,
     3409439.8446457046, 3418782.4480931927, 3416505.47217355, 3427687.473572595, 3472268.805036965, 3532213.072516041,
     3566523.83292695, 3566150.599418379, 3464746.379887561, 3612955.503326156, 3646539.8159785033, 4201940.332113415,
     4365159.896329012, 4326684.855173907, 4519108.935384813, 4490210.022079725, 4972792.2415684555, 5163168.356378808,
     5644870.015958954, 6053609.558716268, 6959190.8205329925, 7710155.234398439, 8377483.872747034, 9163804.634583028,
     9837481.655274473, 12788650.594061475, 14196492.372428989, 20080387.259086467, 544042118.6166384,
     7880072564.352095, 53102743451.985054, 163495831487.94046, 459381177353.12634, 264314996838.1194,
     166398188805.15753, 237188523597.93433, 306428539422.88995, 277455461131.5048, 234155174141.85083,
     90687078788.41716, 146161059193.48926, 234835189364.4999, 281018274612.2062, 466359037699.9051, 405616954703.1383,
     426960265010.1287, 735020735182.3759, 20036239705128.285, 22033890674315.52])


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
    # independientemente del comando
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

    # Ahora se maneja una especie de sistema de puntos para encontrar el que tuvo menor diferencia
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
# a = b = c = d = 0
#
# for i in range(0, len(commands)):
#    print("_______________" + commands[i].upper() + "_______________")
#    for j in range(1, n + 1):
#        file = folder + commands[i] + str(j) + ".wav"
#        ffts = calc.calculate_fft_record(file)  # se calcula la transformada de fourier de la grabación
#
#        parts = calc.split(ffts, nparts)
#
#        energy_sequence = []
#        for k in range(0, nparts):
#            energy_sequence.append(calc.calculate_energy(parts[k]))  # se calcula y guarda la energía de la parte uno
#            # del vector
#            # de la transformada de fourier de la grabación
#
#        command = find_command(energy_sequence)
#
#        if command == "arriba":
#            a = a + 1
#        elif command == "abajo":
#            b = b + 1
#        elif command == "derecha":
#            c = c + 1
#        elif command == "izquierda":
#            d = d + 1
#
#    print("arriba: " + str(a))
#    print("abajo: " + str(b))
#    print("derecha: " + str(c))
#   print("izquierda: " + str(d))
#    a = b = c = d = 0
