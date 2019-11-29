import math as m

numbers = list(map(int, input().split()))
N = numbers[0]
M = numbers[1]
dataset_description = []
for i in range(N):
    dataset_description.append(list(map(int, input().split())))

request_object = list(map(int, input().split()))

function_name = str(input())
# manhattan, euclidean, chebyshev

kernel_function_name = str(input())
# uniform, triangular, epanechnikov, quartic, triweight, tricube, gaussian, cosine, logistic, sigmoid.

window_type_name = str(input())
# fixed — окно фиксированной ширины
# variable — окно переменной ширины

h_or_k = int(input())
# целое число ℎ (0≤ℎ≤100) — радиус окна фиксированной ширины,
# либо целое число 𝐾 (1≤𝐾<𝑁) — число соседей учитываемое для окна переменной ширины.

# N = 3
# M = 2
# dataset_description = [[0, 2, 1], [1, 1, 0], [2, 0, 1]]
# request_object = [0, 0]  # классифицируемый объект
# function_name = 'euclidean'
# kernel_function_name = 'uniform'
# window_type_name = 'fixed'
# h_or_k = 2

# N = 3
# M = 2
# dataset_description = [[0, 2, 1], [1, 1, 0], [2, 0, 1]]
# request_object = [0, 0]  # классифицируемый объект
# function_name = 'euclidean'
# kernel_function_name = 'gaussian'
# window_type_name = 'variable'
# h_or_k = 2


def manhattan(obj):
    res = 0
    for i in range(M):
        res += abs(request_object[i] - dataset_description[obj][i])
    return res


def euclidean(obj):
    res = 0
    for i in range(M):
        res += (request_object[i] - dataset_description[obj][i]) ** 2
    return res ** 0.5


def chebyshev(obj):
    maximum = 0
    for i in range(M):
        res = abs(request_object[i] - dataset_description[obj][i])
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


p = []
for i in range(N):
    p.append([i, distance_function(i)])

num = 0.0
denom = 0.0
delta = 1e-8
w = []

if window_type_name == 'fixed':
    h = h_or_k
    for i in range(N):
        if h != 0:
            w += [kernel_function(p[i][1] / h)]
        else:
            if p[i][1] == 0:  # проходит тест 18, но не проходит 17
                w += [delta]
            else:
                w += [0]
            # w += [delta]  # проходит тест 17, но не проходит 18
    for i in range(N):
        num += dataset_description[i][M] * w[p[i][0]]
        denom += w[i]
    if denom != 0:
        alpha = num / denom
    else:
        for i in range(N):
            num += dataset_description[i][M]
        alpha = num / N

else:
    p = sorted(p, key=lambda x: x[1])
    k = h_or_k
    for i in range(N):
        if p[k][1] == 0:
            if p[i][1] == 0:
                w += [delta]
            else:
                w += [0]
        else:
            w += [kernel_function(p[i][1] / p[k][1])]
    for i in range(N):  # вроде как тут должно быть до k и гауссиана обрезана?
        num += dataset_description[p[i][0]][M] * w[i]
        denom += w[i]
    if denom != 0:
        alpha = num / denom
    else:
        for i in range(N):
            num += dataset_description[i][M]
        alpha = num / N

print(alpha)
