import numpy as np
import calculate as calc

folder = "train_data/"
commands = ["UP", "DOWN", "RIGHT", "LEFT"]
num_parts = 100  # numero de partes en las que se dividieron las grabaciones en el entrenamiento

# Vectores de secuencia de energia promedio de cada comando
arriba = np.array(
    [17605553117512.586, 27145370854283.246, 21678658249812.8, 12670103245338.63, 4875709616423.24, 2230394093377.0127,
     2070960333408.2717, 670832001487.4419, 1100676604322.9192, 1019861704203.5375, 489717208064.24384,
     216271636945.7892, 92298942614.26753, 105497432631.77559, 219150296019.1946, 112976332894.99997,
     28468247137.371754, 6194691891.677486, 2848059454.320225, 4356025196.077885, 4770680397.941829, 5485878220.5883255,
     4007300286.578473, 3109352565.4324093, 2741093379.734898, 2546711095.237618, 3222048537.331538, 2106539267.9828043,
     1064651049.7386593, 1583254211.9515846, 1607706502.4694865, 2247935454.7054973, 3247405375.551637,
     2106947798.8155427, 393119450.79673654, 117773526.89242817, 92675228.27679041, 99184362.78851943,
     110368824.33911118, 105430708.3183023, 63364370.21895888, 57078855.407294765, 49148164.87472491,
     45610687.370367825, 51589524.97113284, 43202088.38050866, 37696239.4112383, 32333370.846657213, 31378972.322378553,
     28715054.507713847, 27279543.282312643, 24025054.210153535, 22096835.938135087, 23581872.88450773,
     20137960.640284073, 18670148.219545253, 18944464.801588796, 16826640.869476296, 13931707.551590228,
     14643922.497269748, 13494318.874501765, 11339437.53400208, 11088530.255756626, 10499920.446316017,
     10752036.945407724, 10760184.845300667, 10043655.743226035, 9768462.412052518, 8739044.908418693,
     8586005.277256863, 9178676.801119668, 9741268.437972387, 9446096.696265582, 8301090.30022899, 7562468.938541266,
     7520565.028472662, 7751676.678080851, 7256987.832210754, 7231285.250529232, 6665958.840481165, 6104912.400271083,
     6352165.4168246575, 6570746.900940978, 6139909.071369252, 5934091.767227955, 5501405.40050731, 5077623.750746587,
     5236800.772263519, 5566684.756408093, 5844674.5559106665, 5304845.548791027, 4888832.30189732, 4967671.134757456,
     5136937.291973057, 5174960.269895957, 5477218.183615369, 4968360.3011406325, 4960027.074141289, 5008824.239176189,
     2386678.171131856])

abajo = np.array(
    [7578517161988.84, 10375362267922.88, 20754078520904.246, 32152185422761.945, 12866682105675.352, 6674920987080.858,
     1944374894586.648, 204898162058.71323, 32293664609.31392, 23615726666.663834, 88896275703.26993,
     137552580047.63168, 154334102560.0765, 69818646184.05725, 56230931712.84021, 33335476323.295097,
     10790763806.550716, 2252656377.362759, 3432501348.829143, 8652592910.07357, 6559158251.003079, 5274359293.327628,
     1365421253.7265983, 1264529574.341548, 2519793831.71598, 1942774429.3794534, 1242660489.8311121, 768809012.807802,
     387171624.2572155, 411472147.6075477, 468423856.25634164, 684230222.4121041, 1038638615.7117267, 896673969.137931,
     317898862.1397079, 73699632.24555808, 55745138.70808485, 55799921.68548822, 59927670.03972804, 39347832.77659537,
     32914284.3954131, 28687040.847513307, 26865078.64888686, 22751882.2978856, 22188104.002823923, 19466473.49801449,
     18528196.032475613, 16640013.518338509, 15163696.32295229, 14216740.93190438, 13809407.177418942,
     12048806.116090432, 12273600.99050703, 10492771.572775485, 10162600.281032598, 10176160.808562635,
     9963994.226470789, 9388389.144498339, 9008509.58213731, 8341700.27229579, 7625034.806946454, 7094743.985895565,
     6488182.762003365, 6537020.673562971, 6087415.066724071, 6047054.536700584, 5801101.078773991, 5656306.079489935,
     5386858.146488065, 5187736.424959793, 5158485.370641092, 4771212.933375546, 4783213.650305674, 4445961.844657812,
     4329475.771747058, 4385284.076837643, 4343110.496899933, 4309311.276763039, 4087127.105558462, 4031071.8696488985,
     3983410.703138582, 3958341.4695766256, 3925422.6361602824, 3861202.045831243, 3796917.77067219, 3812955.68047513,
     3915220.4779833555, 3763491.2313558827, 3758928.6060940996, 3453008.1838262226, 3527403.9626492164,
     3452910.884567895, 3493986.8523581657, 3435574.9330442254, 3520533.358004652, 3517366.887727961, 3525736.361614675,
     3420923.4925126005, 3359712.390114369, 1654532.967412])

derecha = np.array([20429130912243.496, 36660250768742.79, 27959933951029.05, 1318591675657.9688, 634289221465.7987,
                    119174302443.79964, 148387211363.5428, 951105417256.5354, 1417304235305.1953, 552441377781.0615,
                    612792342914.1315, 637033493603.1108, 258770532142.773, 113206071563.3556, 230986712312.4397,
                    528583103459.7571, 494055195952.45856, 159357477762.68814, 40656345969.49728, 103607685860.14651,
                    65528764021.48695, 54715227896.871506, 26795123201.745556, 20823123128.19468, 22500239981.95508,
                    27957318877.09176, 23989617460.293373, 20279275291.971577, 11237139303.178423, 9270214523.141314,
                    21617323022.13925, 43746167843.8501, 52274965745.5077, 24447525277.518505, 4843401685.872528,
                    945052475.8723922, 462776992.2280498, 499756375.1889044, 537197110.1146674, 187207747.16147083,
                    44732710.98705184, 20082101.162829496, 18404683.62614753, 17454306.803040102, 16696213.774861421,
                    14667931.013044015, 12765338.61018076, 11533400.64470155, 11283123.708934505, 10204140.937891643,
                    9199714.35787596, 8600609.210707843, 7964928.726317291, 6944385.0031100325, 6071478.386126385,
                    6048573.747062923, 5515302.08950662, 5752851.000412581, 5631827.3815106135, 5414158.277420287,
                    4994061.08872718, 4990386.065061962, 4806973.825266908, 4890670.377713823, 4771073.597915684,
                    4571298.029588866, 4415721.265536036, 4262574.25650067, 4222508.4665195085, 3974366.9101045355,
                    3895775.9443382225, 3807618.6979046664, 3800856.4427511995, 3735611.329599639, 3784891.468184219,
                    3725613.3494249973, 3711337.245930144, 3652033.950641982, 3608348.6043789648, 3595330.949884282,
                    3550128.0662472337, 3632675.222842186, 3484387.322376228, 3415352.7040807977, 3209628.2977640317,
                    3314615.768297025, 3077970.848537148, 3219391.155319142, 3106860.232005628, 3232218.650616132,
                    3123150.160844779, 3295848.1800939003, 3228588.3744487152, 3318181.106333871, 3198284.5764172855,
                    3188605.088224529, 3141415.1098833685, 3107004.3078199597, 3186630.4946560105, 1730842.1783571595])

izquierda = np.array(
    [18222784501225.9, 38964245742015.055, 24906185390552.2, 2462794429626.452, 731682092171.1138, 286606802485.4752,
     517774433582.6274, 386178166168.76935, 496058411573.5599, 210650495070.63254, 529064112798.019, 472266652631.6816,
     284160263295.77527, 217837513079.33228, 248123471917.39062, 208532431863.2909, 146544343502.10297,
     83120899746.59795, 84487610572.09433, 155934138059.59412, 231966278111.4888, 261470212831.01776, 229793063278.1949,
     445223469028.36646, 202704309118.0747, 286887790259.6714, 247102942554.05896, 188248517060.69092,
     193382754655.70914, 161858898905.785, 244106106096.09402, 473700963829.33563, 462277426993.6178,
     290375689016.41174, 144918670440.56186, 98770546692.24632, 48375559608.964584, 19339335391.583656,
     6517955061.683027, 2282063614.929086, 279947341.8651553, 27937164.47673727, 20250592.09017211, 16286089.703736408,
     13676781.794269994, 13326138.472977817, 13223071.844863007, 10941836.363349097, 9868521.703030461,
     9049903.573584758, 9227255.410670726, 9091384.004255809, 8668670.69326184, 8070423.9928446, 7492423.5902976515,
     7277832.650972479, 7068012.7907823725, 6446644.294610016, 5901055.3192208065, 5944548.684063151, 5588498.622904978,
     5402576.444760498, 5142938.163951253, 4987380.105607816, 4991848.994094261, 4598265.4196104435, 4615806.96608933,
     4464287.631982252, 4611297.315504776, 4478954.560375205, 4281832.03223296, 4394182.414586854, 4314230.320063958,
     4391484.050374713, 4077135.423584899, 3821995.7823003856, 3612441.025283485, 3645029.6423343634, 3551940.180366848,
     3590311.2893053335, 3502981.336714616, 3485689.9486710057, 3591814.895619224, 3574262.1189441043,
     3581324.3381921244, 3526723.656638908, 3560384.243096822, 3465560.5092612035, 3470154.3370702984,
     3426858.5714232926, 3399628.6761694136, 3474713.4995805137, 3369193.4917372274, 3395849.213390553,
     3373555.242411635, 3487105.314256621, 3404744.028054725, 3432614.3624775126, 3492626.6004397995,
     2030017.5436577853])


# FUNCIÓN PARA ENCONTRAR EL COMANDO EN BASE A LA ENERGIA
#
# Se restan la energia promedio de cada comando con la de la que se capta en tiempo real
# Para ver cual tiene menor diferencia
# differences = []
# up = abs(arriba - energy_seq)
# down = abs(abajo - energy_seq)
# right = abs(derecha - energy_seq)
# left = abs(izquierda - energy_seq)
#
# # se separan las diferencias por parte
# # Por ejemplo si tuvieramos estas diferencias:
# # up = [1,2,3,4]
# # down = [5,6,7,8]
# # right = [9,1,2,3]
# # left = [4,5,6,7]
# # Entonces este for haria lo siguiente:
# # differences = [[1,5,9,4],[2,6,1,6],[3,7,2,7],[4,8,3,7]]
# # Es decir que crea un vector que tiene dentro vectores con las diferencias de cada parte
# # independientemente del comando
# for i3 in range(0, num_parts):
#     difference = [up[i3], down[i3], right[i3], left[i3]]
#     differences.append(difference)
#
# # luego con el vector que se creo se busca cual fue el menor de cada parte
# # siguiendo con el ejemplo anterior:
# # differences = [[1,5,9,4],[2,6,1,6],[3,7,2,7],[4,8,3,7]]
# # se busca el menor de cada uno y se guarda la posicion,
# # para la primera parte el menor es 1 por eso se guardaria 0 que es la posicione en la que esta,
# # entonces quedaria asi:
# # min_differences = [0,2,2,2]
# min_differences = []
# for i4 in range(0, 4):
#     index = differences[i4].index(min(differences[i4]))
#     min_differences.append(index)
#
# # Ahora se maneja una especie de sistema de puntos para encontrar el que tuvo menor diferencia
# # Para esto entonces se cuenta cuantas veces aparece una posicion
# # En el caso del ejemplo, ya que tenemos 4 posiciones entonces contamos cuantas veces estan
# # 0, 1, 2, 3 en el vector min_differences:
# # count = [1, 0, 3, 0]
# count = []
# for i5 in range(0, 4):
#     count.append(min_differences.count(i5))
#
# # Luego se mira quien fue el que tuvo mas "puntos" y se toma ese como el comando que probablemente
# # se dijo
# return commands[count.index(max(count))]

def find_command(energy_seq):
    differences = []
    up = abs(arriba - energy_seq)
    down = abs(abajo - energy_seq)
    right = abs(derecha - energy_seq)
    left = abs(izquierda - energy_seq)
    differences.append(np.sum(up))
    differences.append(np.sum(down))
    differences.append(np.sum(right))
    differences.append(np.sum(left))
    min_difference = differences.index(min(differences))
    return commands[min_difference]
