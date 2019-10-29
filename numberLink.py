from dwave_qbsolv import QBSolv
import numpy as np
import collections

grid = [[0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]]
h = {0:-1000, 15:-1000}
J = {(0,0):-1000, (15,15):-1000, (1,1):-1, (4,4):-1, (1,4):2, (11,11):-1, (14,14):-1,(11,14):2}
for x in range(1, 15):
    i = x // 4
    j = x % 4
    if (i*j == 0 and i+j ==3) or i*j == 9:
        if i == 0 and j == 0:
            a = grid[i][j + 1]
            b = grid[i + 1][j]
        elif i == 0 and j == 3:
            a = grid[i][j - 1]
            b = grid[i + 1][j]
        elif i == 3 and j == 0:
            a = grid[i][j + 1]
            b = grid[i - 1][j]
        elif i == 3 and j == 3:
            a = grid[i][j - 1]
            b = grid[i - 1][j]
        near = [a, b]
    elif i == 0 or j == 0:
        if i == 0:
            a = grid[i][j-1]
            b = grid[i][j+1]
            c = grid[i+1][j]
        else:
            a = grid[i][j + 1]
            b = grid[i - 1][j]
            c = grid[i + 1][j]
        near = [a, b, c]
    elif i == 3 or j == 3:
        if i == 3:
            a = grid[i][j-1]
            b = grid[i][j+1]
            c = grid[i-1][j]
        else:
            a = grid[i][j - 1]
            b = grid[i - 1][j]
            c = grid[i + 1][j]
        near = [a, b, c]
    else:
        print((i,j))
        a = grid[i][j - 1]
        b = grid[i][j + 1]
        c = grid[i - 1][j]
        d = grid[i + 1][j]
        near = [a, b, c, d]
    #for ele in near:
    #    if ele in h:
    #        h[ele] -= 4
    #    else:
    #        h[ele] = -4
    for ele1 in near:
        for ele2 in near:
            if ele1 == ele2:
                if(ele1, ele2) in J:
                    J[(ele1, ele2)] -= 3
                else:
                    J[(ele1, ele2)] = -3
            else:
                if ele1 > ele2:
                    continue
                else:
                    if (ele1, ele2) in J:
                        J[(ele1, ele2)] += 2
                    else:
                        J[(ele1, ele2)] = 2

print(h)
print(J)
res = QBSolv().sample_qubo(J)
print(list(res.samples()))
print(list(res.data_vectors['energy']))
for result in list(res.samples()):
    output = list(result.values())
    print(np.array([output[0:4],
           output[4:8],
           output[8:12],
           output[12:16]]))
