K = int(input())
cm = []
for i in range(K):
    cm.append(list(map(int, input().split())))
#
# K = 2
# cm = [[0, 1], [1, 3]]
# K = 3
# cm = [[3, 1, 1], [3, 1, 1], [1, 3, 1]]
amount = 0
rec = []
pr = []
weightRow = []


def f_score(prec, rec):
    if prec + rec != 0:
        return 2 * prec * rec / (prec + rec)
    else:
        return 0


for i in range(K):
    rowAmount = 0
    columnAmount = 0
    for j in range(K):
        amount += cm[i][j]
        rowAmount += cm[i][j]
        columnAmount += cm[j][i]
    weightRow += [rowAmount]
    if columnAmount > 0:
        pr += [cm[i][i] / columnAmount]
    else:
        pr += [0]
    rec += [cm[i][i]]

precision = 0
recall = 0
microF = 0
for i in range(K):
    if amount > 0:
        recall += rec[i] / amount
        precision += pr[i] * weightRow[i] / amount
        microF += f_score(pr[i] * weightRow[i] / amount, rec[i] / amount)

macroF = f_score(precision, recall)

print(round(macroF, 9))
print(round(microF, 9))
