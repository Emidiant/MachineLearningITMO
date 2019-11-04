# K = int(input())
# cm = []
# for i in range(K):
#     cm.append(list(map(int, input().split())))
#
# K = 2
# cm = [[0, 1], [1, 3]]
K = 3
cm = [[3, 1, 1], [3, 1, 1], [1, 3, 1]]

precision = 0
recall = 0
amount = 0
microF = 0.0
rec = []
pr = []
weightRow = []
weightColumn = []
for i in range(K):
    rowAmount = 0
    columnAmount = 0
    for j in range(K):
        amount += cm[i][j]
        rowAmount += cm[i][j]
        columnAmount += cm[j][i]
    precision += cm[i][i] / rowAmount
    pr += [cm[i][i] / rowAmount]
    weightRow += [rowAmount]
    recall += cm[i][i] / columnAmount
    weightColumn += [columnAmount]
    rec += [cm[i][i] / columnAmount]
    if (cm[i][i] / rowAmount) + (cm[i][i] / columnAmount) != 0:
        microF += 2 * (cm[i][i] / rowAmount) * (cm[i][i] / columnAmount) / \
                  ((cm[i][i] / rowAmount) + (cm[i][i] / columnAmount))


precision /= K
recall /= K
macroF = 2 * precision * recall / (precision + recall)
microF /= K

print('macroF', macroF)
print('microF', microF)
print('')
print('Result precision', precision)
print('Result recall', recall)


# мы сначала считаем средний Precision и recall, а из них получаем F-меру,
# или мы считаем F-меру для каждого класса, а потом усредняем её.
# Это деление на micro и macro F-меру.

# С другой стороны, каждый раз, когда мы считаем, что-то среднее мы можем просто суммировать по классам
# и делить на число классов, либо считать средневзвешенное, где вес - число объектов данного  класса.
