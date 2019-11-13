import numpy as np
from dwave_qbsolv import QBSolv

from numberLink import sumToN2


def connected(i, j):
    """

    :param i: index of cell i on grid
    :param j: index of cell j on grid
    :return: 0 if i == j, 2 iff i and j are in same row/col with no black in between, -1 otherwise
    """
    if i[0] != j[0] and i[1] != j[1]:
        # not in same row/col
        return -1
    if i[0] == j[0] and i[1] == j[1]:
        # i == j
        return 0
    elif i[0] == j[0]:
        # same row
        a = min(i[1], j[1])
        b = max(i[1], j[1])
        part = grid[i[0], a:b]
        if np.sum(part != 0) == 0:
            return 2
        else:
            return -1
    else:
        # same col
        a = min(i[0], j[0])
        b = max(i[0], j[0])
        part = grid[a:b, i[1]]
        if np.sum(part != 0) == 0:
            return 2
        else:
            return -1


if __name__ == "__main__":
    hight, width = 5, 5
    grid = np.zeros((hight, width))
    # all positions of black cells
    blacks = [(1, 1), (2, 2), (3, 3)]
    # all number cells and its number (x, y, n)
    numbers = [(1, 2, 1), (2, 3, 2)]
    for b in blacks:
        grid[b] = -1
    for n in numbers:
        grid[n[0:2]] = n[2]

    # assign qbits to empty cells
    var = np.ones((hight, width)) * -1
    count = 0
    for i in range(hight):
        for j in range(width):
            if grid[i, j] == 0:
                var[i, j] = count
                count += 1
    J = {}

    print(grid)
    print(var)
    for i in range(hight):
        for j in range(width):
            if grid[i, j] == -1:  # black cell
                continue
            elif grid[i, j] != 0:  # number cell check its empty cell neighbor
                near = []
                if j > 0:
                    neighbor = grid[i, j - 1]
                    if neighbor == 0: near.append(var[i, j])
                if j < width - 1:
                    neighbor = grid[i, j + 1]
                    if neighbor == 0: near.append(var[i, j + 1])
                if i > 0:
                    neighbor = grid[i - 1][j]
                    if neighbor == 0: near.append(var[i - 1, j])
                if i < hight - 1:
                    neighbor = grid[i + 1][j]
                    if neighbor == 0: near.append(var[i + 1, j])
                if len(near) < grid[i, j]:
                    print("Unsolvable! at {}".format((i, j)))
                sumToN2(near, grid[i, j], J, scale=1)
            else:  # empty cell, write independent set condition
                for k in range(hight):
                    for m in range(width):
                        if grid[k, m] == 0:
                            term = [var[i, j], var[k, m]]
                            term.sort()
                            term = tuple(term)
                            c = connected([i, j], [k, m])
                            print((term, c))
                            if c != 0:
                                if term in J:
                                    J[term] += c
                                else:
                                    J[term] = c

    print(J)
    res = QBSolv().sample_qubo(J, num_repeats=1000)
    samples = list(res.samples())
    energy = list(res.data_vectors['energy'])
    print(samples)
    print(energy)
    for i in range(len(samples)):
        result = samples[i]
        output = grid.copy()
        for k in range(count):
            bit = result[k]
            if bit == 1:
                output[np.where(var == k)] += 0.1

        print("energy: {}_____________________________".format(energy[i]))
        print(output)
