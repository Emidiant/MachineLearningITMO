numbers = list(map(int, input().split()))
objects = list(map(int, input().split()))
N = numbers[0]
M = numbers[1]
K = numbers[2]
# objects = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 1, 2]
# N = len(objects)
# M = 4
# K = 3
result = []
alphabet = []
for r in range(K):
    result.append([])

for r in range(M):
    alphabet.append([])

for r in range(N):
    alphabet[objects[r] - 1].append(r + 1)

counter = 0
for i in range(M):
    for j in range(len(alphabet[i])):
        result[counter % K].append(alphabet[i][j])
        counter += 1

for i in range(len(result)):
    print(len(result[i]), end=' ')
    for j in range(len(result[i])):
        if j != len(result[i]) - 1:
            print(result[i][j], end=' ')
        else:
            print(result[i][j])
