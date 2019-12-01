from dwave_qbsolv import QBSolv
import matplotlib.pyplot as plt
import numpy as np
import minorminer
import networkx as nx
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

#from numberLink import sumToN2
def sumToN2(neighbor, target, J, scale):
    for ele1 in neighbor:
        for ele2 in neighbor:
            term = (ele1, ele2)
 #           if (ele1==1) and(ele2==2): print("hahahaha")
            if ele1 == ele2:
                # for binary variable a^2 = a, thus a^2 - 2*target*a = -(2*target -1)a
                weight = -2 * target + 1
            elif ele1 > ele2:
                continue
            else:
                # 2ab term
                weight = 2 
            if term in J:
                J[term] += weight * scale
            else:
                J[term] = weight * scale



def connected(i, j):
    """

    :param i: index of cell i on grid
    :param j: index of cell j on grid
    :return: 0 if i == j, 2 iff i and j are in same row/col with no black in between, -1 otherwise
    """
    if i[0] != j[0] and i[1] != j[1]:
        # not in same row/col
        #return -3
         return -2
    if i[0] == j[0] and i[1] == j[1]:
        # i == j
        return 0
    elif i[0] == j[0]:
        # same row
        a = min(i[1], j[1])
        b = max(i[1], j[1])
        part = grid[i[0], a:b]-9
        if np.sum(part != 0) == 0:
            #return 50
            return 150
        else:
            #return -5
            return -5
    else:
        # same col
        a = min(i[0], j[0])
        b = max(i[0], j[0])
        part = grid[a:b, i[1]]-9
        if np.sum(part != 0) == 0:
            #return 50
            return 150
        else:
            #return -5
            return -5


if __name__ == "__main__":
    hight, width = 7,7
    grid = np.zeros((hight, width))
    cc=grid
    grid=grid+9
    # all positions of black cells
    blacks = [(0,6),(1,3),(2,0),(5,5),(5,6),(6,1)]
    numbers = [(0,1,0),(2,4,3),(3,3,1),(4,1,4),(5,2,2),(6,5,1)]
   # blacks =[(0,0),(2,1),(2,9),(3,4),(3,9),(4,9),(5,0),(5,1),(8,4),(9,2)]
   # numbers =[(0,9,2),(1,4,1),(2,7,1),(3,8,1),(4,0,0),(4,5,3),(5,6,1),(6,1,2),(6,3,2),(6,5,1),(7,5,0),(7,9,2),(8,0,0),(9,3,1),(9,5,2)]

    for b in blacks:
        grid[b] = -1
    for n in numbers:
        grid[n[0:2]] = n[2]

    # assign qbits to empty cells
    var = np.ones((hight, width)) * -1
    count = 0
    for i in range(hight):
        for j in range(width):
            if grid[i, j] == 9:
                var[i, j] = count
                count += 1
    J = {}

    print(grid)
    print(var) 
    
    for i in range(hight):
        for j in range(width):
            if grid[i, j] == -1:  # black cell
                continue
            elif grid[i, j] != 9:  # number cell check its empty cell neighbor
                near = []
                if j > 0:
                    neighbor = grid[i, j - 1]
                    if neighbor == 9: near.append(var[i, j - 1])
                if j < width - 1:
                    neighbor = grid[i, j + 1]
                    if neighbor == 9: near.append(var[i, j + 1])
                if i > 0:
                    neighbor = grid[i - 1][j]
                    if neighbor == 9: near.append(var[i - 1, j])
                if i < hight - 1:
                    neighbor = grid[i + 1][j]
                    if neighbor == 9: near.append(var[i + 1, j])
                if len(near) < grid[i, j]:
                    print("Unsolvable! at {}".format((i, j)))
                sumToN2(near, grid[i, j], J, scale=120)#250
 #               print(near)
            else:  # empty cell, write independent set condition
                for k in range(hight):
                    for m in range(width):
                        if grid[k, m] == 9:
                            term = [var[i, j], var[k, m]]
              #              term.sort()
                            term = tuple(term)
                            c = connected([i, j], [k, m])
 #                           print((term, c))
                            if c != 0:
                                if term in J:
                                    J[term] += c
                                else:
                                    J[term] = c
                                
 #   print(J)
use_qpu=True
if use_qpu:
    solver_limit = 36
    G = nx.complete_graph(solver_limit)
    system = DWaveSampler(token='DEV-6189564036d19f88b3a555b4175a353d6d2c0218')
    embedding = minorminer.find_embedding(J.keys(), system.edgelist)
    print(embedding)
    res = QBSolv().sample_qubo(J, solver=FixedEmbeddingComposite(system, embedding), solver_limit=solver_limit, num_reads=3000)
    #Emb = EmbeddingComposite(DWaveSampler(token='DEV-6189564036d19f88b3a555b4175a353d6d2c0218'))
    #res = Emb.sample_qubo(D, num_reads=10000)
else:
    res = QBSolv().sample_qubo(J, num_repeats=3000)
samples = list(res.samples())
energy = list(res.data_vectors['energy'])
#    print(samples)
#    print(energy)
for i in range(len(samples)):
    result = samples[i]
    output = grid.copy()
    for k in range(count):
        bit = result[k]
        if bit == 1:
            output[np.where(var == k)]+=5
    if i<5: 
        print("energy: {}_____________________________".format(energy[i]))
    if i<5:
        print(output)
    
 #       for j in range(hight):
  #          for m in range(width):
   #             if output[j, m] == 0: 
    #            elif output[j, m] == 5: 
     #                   cc[j, m] ='L'
      #          elif output[j,m]==-1: cc[j,m]='B'
       #         else: cc[j, m] = str(output[j,m])