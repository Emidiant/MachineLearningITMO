import pandas as pd
import matplotlib.pyplot as plt
import math as m
import numpy
import matplotlib as mpl
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 5)


def read_lines(filename):
    with open(filename) as f:
        return len(f.readlines())


def f_score(prec, rec):
    if prec + rec != 0:
        return 2 * prec * rec / (prec + rec)
    else:
        return 0


def manhattan(obj):
    res = 0
    for i in range(M):
        res += abs(request_object[i] - train_sample[obj][i])
    return res


def euclidean(obj):
    res = 0
    for i in range(M):
        res += (request_object[i] - train_sample[obj][i]) ** 2
    return res ** 0.5


def chebyshev(obj):
    maximum = 0
    for i in range(M):
        res = abs(request_object[i] - train_sample[obj][i])
        if res > maximum:
            maximum = res
    return maximum


def distance_function(obj):
    return {
        'manhattan': manhattan(obj),
        'euclidean': euclidean(obj),
        'chebyshev': chebyshev(obj)
    }.get(function_name)


def kernel_function(u):
    if abs(u) >= 1:
        return {
            'gaussian': 1 / m.sqrt(2 * m.pi) * m.e ** (- 0.5 * u ** 2),
            'logistic': 1 / (m.e ** u + 2 + m.e ** (-u)),
            'sigmoid': 2 / (m.pi * (m.e ** u + m.e ** (-u)))
        }.get(kernel_function_name, 0)
    else:
        return {
            'uniform': 0.5,
            'triangular': 1 - abs(u),
            'epanechnikov': 0.75 * (1 - u ** 2),
            'quartic': 15 / 16 * (1 - u ** 2) ** 2,
            'triweight': 35 / 32 * (1 - u ** 2) ** 3,
            'tricube': 70 / 81 * (1 - abs(u) ** 3) ** 3,
            'gaussian': 1 / m.sqrt(2 * m.pi) * m.e ** (- 0.5 * u ** 2),
            'cosine': m.pi / 4 * m.cos(m.pi / 2 * u),
            'logistic': 1 / (m.e ** u + 2 + m.e ** (-u)),
            'sigmoid': 2 / (m.pi * (m.e ** u + m.e ** (-u)))
        }.get(kernel_function_name)


data = pd.read_csv('data/zoo.csv')
N = read_lines('data/zoo.csv') - 2

columns = []
for i in data:
    columns.append(i)
M = len(columns) - 2
print(columns)
print(N, 'объект')
print(M, 'признаков')
print(M + 1, 'вид животного')

types = []
classes = []
for i in range(N):
    value = data[columns[M + 1]][i]
    types.append(value)
    if value not in classes:
        classes.append(value)
classes.sort()

dict_classes = {}
for i in range(len(classes)):
    dict_classes[classes[i]] = i + 1

for i in range(N):
    types[i] = dict_classes[types[i]]

print(dict_classes)
print(types)
counter = 0
for i in range(len(types)):
    if types[i] == 6:
        counter += 1
print('mammal', counter)

dataset_description = []
for i in range(N):
    dataset_description.append([])
    for j in range(M + 1):
        if str(data[columns[j + 1]][i]) == 'True':
            dataset_description[i].append(1)
        else:
            if str(data[columns[j + 1]][i]) == 'False':
                dataset_description[i].append(0)
            else:
                if columns[j + 1] == 'legs':
                    dataset_description[i].append(data[columns[j + 1]][i] / max(data['legs']))
                else:
                    dataset_description[i].append(types[i])

print(dataset_description)

num_classes = len(classes)
delta = 1e-3
accuracy = []
macroF_array = []
microF_array = []
h_or_k_array = []
function_name = 'manhattan'
kernel_function_name = 'gaussian'
window_type_name = 'fixed'
# manhattan, euclidean, chebyshev
# uniform, triangular, epanechnikov, quartic, triweight, tricube, gaussian, cosine, logistic, sigmoid
# fixed, variable
dict_result = {}
train_N = N - 1
if window_type_name == 'fixed':
    limit = 5
else:
    limit = N - 1
h_or_k = 0
while h_or_k < limit:
    if window_type_name == 'fixed':
        h_or_k += 0.1
    else:
        h_or_k += 1

    h_or_k_array.append(h_or_k)
    alpha_res = []

    counter_right_ans = 0
    confusion_matrix = [[0 for x in range(num_classes)] for y in range(num_classes)]

    for i in range(train_N):
        train_sample = dataset_description
        request_object = train_sample[i]
        train_sample = numpy.delete(train_sample, i, axis=0)

        p = []
        for k in range(train_N):
            p.append([k, distance_function(k)])

        w = []
        if window_type_name == 'fixed':
            h = h_or_k
            for i in range(train_N):
                w += [kernel_function(p[i][1] / h)]
            for i in range(train_N):
                cur_dict = dict_result.get(train_sample[i][M], 0)
                dict_result[train_sample[i][M]] = cur_dict + w[p[i][0]]
        else:
            p = sorted(p, key=lambda x: x[1])
            if h_or_k == limit:
                k = h_or_k - 1
            else:
                k = h_or_k
            for i in range(train_N):
                if p[k][1] == 0:
                    if p[i][1] == 0:
                        w += [delta]
                    else:
                        w += [0]
                else:
                    w += [kernel_function(p[i][1] / p[k][1])]
            for i in range(train_N):
                cur_dict = dict_result.get(train_sample[p[i][0]][M], 0)
                dict_result[train_sample[p[i][0]][M]] = cur_dict + w[i]

        alpha_res.append([int(max(dict_result, key=dict_result.get)), request_object[M]])

        confusion_matrix[request_object[M] - 1][int(max(dict_result, key=dict_result.get)) - 1] += 1
        dict_result = {}
    for i in range(len(alpha_res)):
        if alpha_res[i][0] == alpha_res[i][1]:
            counter_right_ans += 1
    if window_type_name == 'fixed':
        print('h =', h_or_k)
    else:
        print('k =', h_or_k)

    # F measure
    amount = 0
    rec = []
    pr = []
    weightRow = []

    for i in range(num_classes):
        rowAmount = 0
        columnAmount = 0
        for j in range(num_classes):
            amount += confusion_matrix[i][j]
            rowAmount += confusion_matrix[i][j]
            columnAmount += confusion_matrix[j][i]
        weightRow += [rowAmount]
        if columnAmount > 0:
            pr += [confusion_matrix[i][i] / columnAmount]
        else:
            pr += [0]
        rec += [confusion_matrix[i][i]]

    precision = 0
    recall = 0
    microF = 0
    for i in range(num_classes):
        if amount > 0:
            recall += rec[i] / amount
            precision += pr[i] * weightRow[i] / amount
            microF += f_score(pr[i] * weightRow[i] / amount, rec[i] / amount)

    macroF = f_score(precision, recall)

    print(round(macroF, 9))
    print(round(microF, 9))
    macroF_array.append(round(macroF, 9))
    microF_array.append(round(microF, 9))

print(macroF_array)
print(microF_array)

dpi = 80
fig = plt.figure(dpi=dpi, figsize=(700 / dpi, 500 / dpi))
mpl.rcParams.update({'font.size': 10})

title = 'F-score, ' + str(function_name) + ', ' + str(kernel_function_name)
plt.title(title)

plt.ylabel('F')

if window_type_name == 'fixed':
    plt.xlabel('h * 1e-1')
else:
    plt.xlabel('k')

plt.plot(h_or_k_array, macroF_array, color='blue', linestyle='-', label='macroF')
plt.plot(h_or_k_array, microF_array, color='red', linestyle='-', label='microF')
print(max(macroF_array), max(microF_array))
maxF = 0
hk_opt = []
for i in range(len(macroF_array)):
    if macroF_array[i] == maxF:
        hk_opt += [h_or_k_array[i]]
    if microF_array[i] == maxF:
        hk_opt += [h_or_k_array[i]]
    if macroF_array[i] > maxF:
        hk_opt = []
        maxF = macroF_array[i]
        hk_opt += [h_or_k_array[i]]
    if microF_array[i] > maxF:
        hk_opt = []
        maxF = microF_array[i]
        hk_opt += [h_or_k_array[i]]

print('Best result:', 'hk', hk_opt, 'max F-measure', maxF)

plt.legend(loc='upper right')
plt.grid(True)
fileName = str(function_name) + '_' + str(kernel_function_name) + '_' + str(window_type_name) + '.png'
fig.savefig(fileName)
